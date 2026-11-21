"""Device-side preflight and execution; does not import Inspect Robots or an LLM SDK."""
from collections import deque
import base64
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import tempfile

import numpy as np
from PIL import Image

from .controller import HardwareDriver, MockDriver, RealController, StopRequested
from .kinematics import JOINTS, MOTORS, load_profile, wrap
from .settings import load_settings, execution_gate


class HardwareRuntime:
    def __init__(self, config_path, output, *, mode="mock", assume_calibration=False,
                 max_steps=1800, policy_seconds=900, reference_image=None, video=True,
                 requires_cartesian_verification=True):
        self.settings, raw_profile = load_settings(config_path)
        execution_gate(self.settings, raw_profile, mode, assume_calibration, requires_cartesian=requires_cartesian_verification)
        self.profile = load_profile(self.settings["profile_file"])
        self.profile["calibration_assumed"] = bool(assume_calibration)
        if not isinstance(max_steps, int) or max_steps < 1 or not math.isfinite(policy_seconds) or policy_seconds <= 0:
            raise ValueError("Positive step and wall-time limits are required")
        self.mode, self.max_steps, self.policy_seconds = mode, max_steps, policy_seconds
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = (self.output/"motor_events.jsonl").open("a")
        self.video = None
        self.driver = self.controller = None
        self.serial_lock = None
        self.preview_open = False
        self.preview_enabled = self.settings["preview"] and mode != "mock"
        self.reference = None
        self.reference_image = reference_image
        self.operator_mode = False
        self.operator_label = ""
        self.approved = deque()
        self.plan_id = 0
        self.replay_plan = None
        self.first_waypoint = False
        self.plan_start_motors = None
        self.started = False
        self.started_at = None
        self._holding = False
        self.stopped = None
        self.planned_steps = self.framework_steps = self.reset_steps = self.motor_writes = self.hold_writes = 0
        self.last_video_frame = 0.
        self.last_state = None
        self.last_images = None
        self.feedback = dict(accepted=True, position_error_m=0., orientation_error_rad=0., detail="ready")
        calibration = json.loads(Path(self.settings["calibration_file"]).read_text())
        self.calibration = calibration
        if self.preview_enabled and not os.environ.get("DISPLAY"):
            self.events.close()
            raise ValueError("Hardware preview needs desktop DISPLAY; set preview=false for terminal-only operation")
        try:
            if mode == "mock":
                self.driver = MockDriver(state_file=self.settings.get("mock_state_file"))
            else:
                port_key = hashlib.sha256(os.path.realpath(self.settings["port"]).encode()).hexdigest()[:24]
                self.serial_lock = (Path(tempfile.gettempdir())/f"llm_libero_so101_{port_key}.lock").open("a")
                try:
                    fcntl.flock(self.serial_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError("This serial port is already in use by another LLM_LIBERO hardware worker") from None
                self.driver = HardwareDriver(port=self.settings["port"], follower_id=self.settings["follower_id"],
                    calibration_dir=Path(self.settings["calibration_file"]).parent,
                    top_index=self.settings["top_index"], wrist_index=self.settings["wrist_index"],
                    width=self.settings["width"], height=self.settings["height"], fps=self.settings["fps"], execute=mode == "execute")
            self.controller = RealController(self.driver, self.profile, calibration,
                execute=mode in ("mock", "execute"), mock=mode == "mock", zoom=self.settings["zoom"],
                max_steps=max_steps, stop_file=self.output/"STOP", on_tick=self._capture)
            self.controller.on_motor_action = self._motor_action
            original_send, original_hold = self.driver.send, self.controller.hold
            def counted_send(motors):
                sent = original_send(motors)
                self.controller.last_motor_goal = dict(sent)
                self.controller.last_gripper_motor_goal = sent["gripper.pos"]
                self.motor_writes += 1
                if self._holding or not self.started and not self.reset_steps and self.controller.observed_at is None:
                    self.hold_writes += 1
                self.event("motor_write", stage="hold" if self._holding else "episode" if self.started else "setup",
                           requested=motors, sent=sent)
                return sent
            def counted_hold():
                self._holding = True
                try:
                    return original_hold()
                finally:
                    self._holding = False
            self.driver.send, self.controller.hold = counted_send, counted_hold
            # Validate fixed-reset values against motor travel before opening serial.
            self.reset_target = None
            if self.settings["reset_mode"] == "fixed":
                self.reset_target = json.loads(Path(self.settings["reset_action_file"]).read_text())
                self.controller.valid_motors(self.reset_target)
            self.driver.connect()
            if video and self.settings["video"]:
                import imageio.v2 as imageio
                self.video = imageio.get_writer(str(self.output/"rollout.mp4"), fps=10, codec="libx264", macro_block_size=2)
            self._snapshot(calibration, config_path)
            self.capture_observation()
        except BaseException:
            self.close()
            raise

    def event(self, kind, **values):
        self.events.write(json.dumps(dict(event=kind, monotonic_s=time.monotonic(), **values)) + "\n")
        self.events.flush()

    def _snapshot(self, calibration, config_path):
        files = dict(config=str(Path(config_path).resolve()), profile=self.settings["profile_file"],
                     calibration=self.settings["calibration_file"], urdf=self.profile["urdf"])
        for name in ("runtime.py", "controller.py", "kinematics.py", "settings.py", "embodiment.py", "approver.py", "replay.py", "_worker.py"):
            files["adapter_"+name] = str(Path(__file__).with_name(name))
        if self.settings.get("reset_action_file"):
            files["reset"] = self.settings["reset_action_file"]
        data = dict(mode=self.mode, calibration_assumed=self.profile["calibration_assumed"],
            settings=self.settings, profile=self.profile, calibration=calibration,
            source_files=files, source_sha256={k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in files.items()},
            robot_state_source="encoder forward kinematics; no external pose tracker", task_success=None)
        (self.output/"hardware_snapshot.json").write_text(json.dumps(data, indent=2))

    def _motor_action(self, requested, sent, measured):
        self.event("motor_command", stage="episode" if self.started else "reset",
                   requested=requested, sent=sent, measured_before=measured)

    def diagnostics(self):
        # A separate read-only operation, outside task motion and model tools.
        # Do not stall the timed control loop with many diagnostic register reads.
        report = self.driver.read_diagnostics()
        report.update(device_mode=self.mode, profile_file=self.settings["profile_file"])
        (self.output/"motor_diagnostics.json").write_text(json.dumps(report, indent=2))
        return report

    def hold(self):
        if self.controller and self.controller.execute:
            self.controller.hold()
            self.event("hold", preserves_gripper_target=True)

    def _capture(self, images):
        now = time.monotonic()
        if self.video is not None and now-self.last_video_frame >= .095:
            self.video.append_data(np.concatenate([images["top"], images["wrist"]], axis=1))
            self.last_video_frame = now
        if self.preview_enabled:
            import cv2
            top = images["top"].copy()
            if self.settings["reference_overlay"] and self.reference_image and self.reference is None:
                self.reference = np.asarray(Image.open(self.reference_image).convert("RGB"))
            if self.settings["reference_overlay"] and self.reference is not None:
                overlay = cv2.resize(self.reference, (top.shape[1], top.shape[0]))
                top = cv2.addWeighted(top, .65, overlay, .35, 0)
            if self.operator_label:
                cv2.putText(top,self.operator_label,(8,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,220,40),1,cv2.LINE_AA)
            preview = np.concatenate([top, images["wrist"]], axis=1)
            title = "LLM_LIBERO SO-101: top reference overlay | wrist (q stops)"
            cv2.imshow(title, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            self.preview_open = True
            if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise StopRequested("operator closed preview")

    def pump(self):
        if self.stopped:
            raise StopRequested(self.stopped)
        self.controller.pump()

    def capture_observation(self):
        state, images = self.controller.observe()
        self.last_state, self.last_images = state, images
        self._capture(images)
        return self.observation()

    def observation(self):
        # JSON-compatible numeric robot state plus lossless 256px policy images.
        state = json.loads(json.dumps(self.last_state))
        encoded = {}
        for name, pixels in self.last_images.items():
            image = Image.fromarray(pixels).resize((256, 256))
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            encoded[name] = base64.b64encode(stream.getvalue()).decode()
            frame = state["reference_frames"]["cameras"][name]
            if "K_after_crop_resize" in frame:
                k = np.array(frame["K_after_crop_resize"])
                k[0] *= 256/pixels.shape[1]
                k[1] *= 256/pixels.shape[0]
                frame["K_after_crop_resize"] = k.tolist()
            frame["image_size_px"] = [256, 256]
        return dict(state=state, images_png=encoded, feedback=self.feedback, **self.summary())

    def setup(self):
        if self.reset_target is not None:
            self.controller.max_steps = self.settings["reset_max_steps"]
            result = self.controller.reset(self.reset_target)
            self.reset_steps = self.controller.steps
            self.event("fixed_reset", result=result)
            if result["status"] not in ("executed", "planned_only"):
                raise StopRequested("fixed reset failed")
            if result["status"] == "executed":
                gripper_goal = self.controller.last_gripper_motor_goal
                if gripper_goal is None:
                    gripper_goal = self.reset_target["gripper.pos"]
                self.controller.gripper_command = float(np.clip(
                    (gripper_goal-self.profile["gripper_open_motor"])/
                    (self.profile["gripper_closed_motor"]-self.profile["gripper_open_motor"]), 0, 1))
        self.controller.steps = 0
        self.controller.max_steps = self.max_steps
        return self.capture_observation()

    def start(self):
        self.operator_mode = False
        self.started = True
        self.started_at = time.monotonic()
        self.controller.deadline = time.monotonic()+self.policy_seconds
        return self.capture_observation()

    def operator_view(self, reference_image, label):
        """Untimed human setup/review. No reset or task action is performed."""
        self.approved.clear()
        self.replay_plan = None
        self.started = False
        self.started_at = None
        self.operator_mode = True
        self.controller.deadline = float("inf")
        self.reference_image = reference_image
        self.reference = None
        self.operator_label = label
        self.event("operator_view",reference_image=reference_image,label=label)
        return self.capture_observation()

    def operator_reset(self):
        """One explicit keyboard reset, counted separately from agent actions."""
        if not self.operator_mode:
            raise ValueError("Manual reset is only available in operator setup/review mode")
        if not self.settings.get("reset_action_file"):
            raise ValueError("Operator reset requires reset_action_file")
        target=json.loads(Path(self.settings["reset_action_file"]).read_text())
        ctrl=self.controller
        ctrl.check_stop()
        ctrl.valid_motors(target)
        self.approved.clear()
        self.replay_plan=None
        ctrl.steps=0
        ctrl.max_steps=self.settings["reset_max_steps"]
        ctrl.deadline=time.monotonic()+self.policy_seconds
        try:
            result=ctrl.reset(target)
            self.event("operator_reset",target=target,result=result)
            if result["status"] not in ("executed","planned_only"):
                self.stopped=result.get("reason",result["status"])
            if result["status"]=="executed":
                goal=ctrl.last_gripper_motor_goal
                ctrl.gripper_command=float(np.clip((goal-self.profile["gripper_open_motor"])/
                    (self.profile["gripper_closed_motor"]-self.profile["gripper_open_motor"]),0,1))
            self.feedback=dict(accepted=result["status"] in ("executed","planned_only"),position_error_m=0.,
                               orientation_error_rad=0.,detail="operator_reset:"+result["status"])
        finally:
            self.reset_steps+=ctrl.steps
            ctrl.steps=0
            ctrl.max_steps=self.max_steps
            ctrl.deadline=float("inf")
        return self.capture_observation()

    def _remaining(self):
        spent = self.controller.steps if self.controller.execute else self.planned_steps
        return self.max_steps-spent

    def plan(self, targets):
        """Validate the entire upstream Cartesian chunk before allowing its first motor write."""
        self.approved.clear()
        self.replay_plan = None
        self.controller.check_stop()
        if self.mode == "inspect":
            return dict(accepted=False, reason="Inspection mode has no motion planning/execution")
        array = np.asarray(targets, dtype=float)
        if array.ndim != 2 or array.shape[1] != 6 or not 1 <= len(array) <= self.max_steps or not np.isfinite(array).all():
            return dict(accepted=False, reason="Expected a finite N-by-6 Cartesian path within the step budget")
        ctrl = self.controller
        if ctrl.observed_at is None or time.monotonic()-ctrl.observed_at > self.profile["observation_max_age_s"]:
            return dict(accepted=False, reason="Observation expired; motion blocked")
        current = self.driver.read_motors()
        ctrl.valid_motors(current)
        if max(abs(current[n]-ctrl.observed_motors[n]) for n in MOTORS[:5]) > 3:
            return dict(accepted=False, reason="Robot drifted since observation; motion blocked")
        q = ctrl.kinematics.motor_to_model(current)
        start_pose = ctrl.kinematics.forward(q)
        start_angles = ctrl.kinematics.orientation(start_pose)
        if np.any(q < ctrl.kinematics.limits[:, 0]) or np.any(q > ctrl.kinematics.limits[:, 1]):
            return dict(accepted=False, reason="Start pose is outside model joint limits; check profile/reset")
        low = np.array([self.profile["workspace_m"][axis][0] for axis in ("x", "y", "z")] + [-math.pi, -math.pi, 0])
        high = np.array([self.profile["workspace_m"][axis][1] for axis in ("x", "y", "z")] + [math.pi, math.pi, 1])
        if np.any(array < low) or np.any(array > high):
            return dict(accepted=False, reason="Cartesian path leaves configured workspace/angle/gripper bounds")
        if np.max(np.linalg.norm(array[:, :3]-start_pose[:3, 3], axis=1)) > self.profile["max_translation_m"]+1e-6:
            return dict(accepted=False, reason="Move exceeds configured per-call translation limit")
        speed = np.array([self.profile["joint_speed_deg_s"]]*5 + [self.profile["gripper_speed_units_s"]])
        previous_goal = ctrl.last_motor_goal or current
        previous = np.array([previous_goal[n] for n in MOTORS])
        # A grasp can stop the finger short of its commanded closure. Retain
        # that closure target instead of reopening toward the measured gap.
        if ctrl.last_gripper_motor_goal is not None:
            previous[-1] = ctrl.last_gripper_motor_goal
        planned, segments = 0, []
        for target in array:
            ctrl.check_stop()
            # Preserve the existing arm servo goals for truly gripper-only chunks.
            pose_change = np.linalg.norm(target[:3]-start_pose[:3, 3]) > 1e-7 or np.max(np.abs(wrap(target[3:5]-start_angles))) > 1e-7
            if pose_change:
                q, pe, re = ctrl.kinematics.solve(target[:3], target[3:5], q, np.array([True, True]), ctrl.motor_limits)
                if pe > self.profile["position_tolerance_m"] or re > .1:
                    return dict(accepted=False, reason=f"No feasible IK path (position {pe:.4f} m; angle {re:.3f} rad)")
            gripper = self.profile["gripper_open_motor"] + target[5]*(self.profile["gripper_closed_motor"]-self.profile["gripper_open_motor"])
            motors = ctrl.kinematics.model_to_motor(q, gripper)
            vector = ctrl.valid_motors(motors)
            count = max(1, math.ceil(float(np.max(np.abs(vector-previous)/(speed*ctrl.dt)))))
            planned += count
            if planned > self._remaining():
                return dict(accepted=False, reason="Preflight motor path exceeds remaining control-step budget")
            path = [dict(zip(MOTORS, (previous+fraction*(vector-previous)).tolist())) for fraction in np.linspace(1/count, 1, count)]
            for motor_target in path:
                predicted = ctrl.kinematics.forward(ctrl.kinematics.motor_to_model(motor_target))[:3, 3]
                if np.any(predicted < low[:3]-1e-6) or np.any(predicted > high[:3]+1e-6):
                    return dict(accepted=False, reason="Interpolated motor path leaves the configured Cartesian workspace")
            segments.append(dict(target=target.tolist(), path=path))
            previous = vector
        if time.monotonic()-ctrl.observed_at > self.profile["observation_max_age_s"]:
            return dict(accepted=False, reason="Observation expired during IK preflight")
        fresh = self.driver.read_motors()
        ctrl.valid_motors(fresh)
        if max(abs(fresh[n]-current[n]) for n in MOTORS[:5]) > 2:
            return dict(accepted=False, reason="Robot moved during IK preflight")
        self.plan_id += 1
        self.first_waypoint = True
        self.plan_start_motors = dict(current)
        self.approved.extend(segments)
        self.event("preflight", plan_id=self.plan_id, motor_steps=planned, segments=segments)
        return dict(accepted=True, plan_id=self.plan_id, motor_steps=planned)

    def step(self, target, plan_id):
        self.controller.check_stop()
        target = np.asarray(target, dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all() or not self.approved or plan_id != self.plan_id or not np.allclose(target, self.approved[0]["target"], rtol=0, atol=1e-7):
            self.event("waypoint_rejected", reason="no matching preflight", received_plan_id=plan_id,
                       approved_plan_id=self.plan_id, expected_target=self.approved[0]["target"] if self.approved else None,
                       received_target=target.tolist() if target.shape == (6,) and np.isfinite(target).all() else "invalid")
            self.approved.clear()
            self.feedback = dict(accepted=False, position_error_m=0., orientation_error_rad=0., detail="Waypoint has no matching whole-path preflight")
            return self.capture_observation()
        if self.first_waypoint:
            fresh = self.driver.read_motors()
            self.controller.valid_motors(fresh)
            expired = time.monotonic()-self.controller.observed_at > self.profile["observation_max_age_s"]
            drifted = max(abs(fresh[n]-self.plan_start_motors[n]) for n in MOTORS[:5]) > 2
            if expired or drifted:
                self.approved.clear()
                self.feedback = dict(accepted=False, position_error_m=0., orientation_error_rad=0., detail="Observation expired or robot moved after preflight; no motion sent")
                return self.capture_observation()
            self.first_waypoint = False
        segment = self.approved.popleft()
        self.planned_steps += len(segment["path"])
        result = self.controller.execute_path(segment["path"])
        self.framework_steps += 1
        if result["status"] == "executed":
            self.controller.gripper_command = float(target[5])
        if result["status"] not in ("executed", "planned_only"):
            self.approved.clear()
            self.stopped = result.get("reason", result["status"])
            if result["status"] != "tracking_failed":
                self.hold()  # Tracking failures already requested a hold in execute_path.
        self.feedback = dict(accepted=result["status"] in ("executed", "planned_only"),
                             position_error_m=0., orientation_error_rad=0., detail=result["status"])
        self.event("waypoint_result", result=result)
        observed = self.capture_observation()
        measured = observed["state"]
        self.feedback["position_error_m"] = float(np.linalg.norm(np.asarray(measured["eef_position_m"])-np.asarray(target)[:3]))
        self.feedback["orientation_error_rad"] = float(np.max(np.abs(wrap(np.array([measured["pitch_rad"], measured["roll_rad"]])-np.asarray(target)[3:5]))))
        observed["feedback"] = self.feedback
        return observed

    def prepare_replay(self, targets, *, source_hz=20, time_scale=2., source_start_step=1):
        """Preflight a complete recording from fresh physical state, without motion."""
        from .replay import plan_replay, ReplayRejected
        self.replay_plan = None
        self.approved.clear()
        if self.mode == "inspect":
            return dict(accepted=False, reason="Inspection mode cannot prepare motion")
        self.controller.check_stop()
        observed_at = time.monotonic()
        current = self.driver.read_motors()
        self.controller.valid_motors(current)
        try:
            plan = plan_replay(self.profile, self.calibration, targets, current,
                source_hz=source_hz, time_scale=time_scale, max_steps=self._remaining(), source_start_step=source_start_step,
                check_stop=self.controller.check_stop)
            if time.monotonic()-observed_at > self.profile["observation_max_age_s"]:
                raise ReplayRejected("Observation expired while preparing the replay")
            self.plan_id += 1
            plan.update(plan_id=self.plan_id, observation_monotonic_s=observed_at)
            self.replay_plan = plan
            (self.output/"replay_plan.json").write_text(json.dumps(plan))
            result = dict(accepted=True, plan_id=self.plan_id, **plan["summary"])
        except ReplayRejected as exc:
            result = exc.report
        (self.output/"replay_preflight.json").write_text(json.dumps(result,indent=2))
        self.event("replay_preflight", result=result)
        return result

    def run_replay(self, plan_id):
        """Execute only a complete cached plan; keep all controller stop guards."""
        plan, self.replay_plan = self.replay_plan, None  # Single-use, including rejected attempts.
        ctrl = self.controller
        if plan is None or plan_id != plan["plan_id"]:
            return dict(accepted=False, reason="Replay has no matching full-path preflight")
        ctrl.check_stop()
        current = self.driver.read_motors()
        ctrl.valid_motors(current)
        if time.monotonic()-plan["observation_monotonic_s"] > self.profile["observation_max_age_s"]:
            return dict(accepted=False, reason="Replay start observation expired; prepare again")
        if max(abs(current[n]-plan["start_motors"][n]) for n in MOTORS) > 2:
            return dict(accepted=False, reason="Robot moved after replay preflight; prepare again")
        self.planned_steps += len(plan["motor_targets"])
        if self.mode == "dry-run":
            return dict(accepted=True, status="planned_only", **plan["summary"])
        started = time.monotonic()
        peak_error = 0.
        result = dict(status="executed")
        completed = 0
        columns = ["motor_step","source_step","elapsed_s","max_arm_error_deg","eef_error_m"]
        columns += [f"{label}:{motor}" for label in ("commanded","measured_after") for motor in MOTORS]
        with (self.output/"replay_tracking.csv").open("w",newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            try:
                for index, vector in enumerate(plan["motor_targets"]):
                    motors = dict(zip(MOTORS,vector))
                    result = ctrl.execute_path([motors])
                    self.event("replay_waypoint", motor_step=index+1,source_step=plan["source_steps"][index],result=result)
                    if result["status"] != "executed":
                        self.stopped = result.get("reason",result["status"])
                        if result["status"] != "tracking_failed":
                            self.hold()
                        break
                    completed += 1
                    actual = self.driver.read_motors()
                    ctrl.valid_motors(actual)
                    error = max(abs(actual[n]-motors[n]) for n in MOTORS[:5])
                    peak_error = max(peak_error,error)
                    wanted_pose = ctrl.kinematics.forward(ctrl.kinematics.motor_to_model(motors))
                    actual_pose = ctrl.kinematics.forward(ctrl.kinematics.motor_to_model(actual))
                    position_error = float(np.linalg.norm(wanted_pose[:3,3]-actual_pose[:3,3]))
                    writer.writerow([index+1,plan["source_steps"][index],time.monotonic()-started,error,position_error]
                        +vector+[actual[n] for n in MOTORS])
            except BaseException as exc:
                self.stopped = str(exc) or type(exc).__name__
                try:
                    self.hold()
                finally:
                    result = dict(status="stopped",reason=self.stopped)
                raise
            finally:
                report = dict(accepted=True,status="replayed" if completed == len(plan["motor_targets"]) else "hardware_stopped",
                    completed_motor_steps=completed,elapsed_s=time.monotonic()-started,peak_arm_tracking_error_deg=peak_error,
                    stop_detail=self.stopped,last_controller_result=result,device_mode=self.mode,
                    real_api_calls=0,agent_cost_usd=0,physical_success=None,
                    eef_error_source="Encoder FK in hardware profile frame; no external pose measurement")
                (self.output/"replay_summary.json").write_text(json.dumps(report,indent=2))
        return report

    def summary(self):
        return dict(steps=self.controller.steps if self.controller else 0, framework_steps=self.framework_steps,
                    planned_control_steps=self.planned_steps, reset_control_steps=self.reset_steps,
                    motor_writes=self.motor_writes, hold_writes=self.hold_writes, device_mode=self.mode,
                    policy_elapsed_s=time.monotonic()-self.started_at if self.started_at is not None else None,
                    stopped=self.stopped, environment_success=None, human_success=None)

    def close(self):
        if self.events.closed:
            return
        try:
            if self.controller is not None:
                try:
                    self.hold()
                except Exception as exc:
                    self.event("hold_failed", error_type=type(exc).__name__)
            if self.video is not None:
                self.video.close()
                self.video = None
            if self.preview_open:
                import cv2
                cv2.destroyAllWindows()
                self.preview_open = False
        finally:
            try:
                if self.driver is not None:
                    self.driver.close()
            finally:
                if self.serial_lock is not None:
                    self.serial_lock.close()
                    self.serial_lock = None
                (self.output/"hardware_summary.json").write_text(json.dumps(self.summary(), indent=2))
                self.events.close()

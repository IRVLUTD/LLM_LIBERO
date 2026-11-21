"""Hardware embodiment for the unchanged Inspect Robots LLM agent policy."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np
from PIL import Image
from inspect_robots.embodiment import EmbodimentBase, EmbodimentInfo
from inspect_robots.errors import ConfigError, EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec
from inspect_robots.task import Task
from inspect_robots.types import Observation, StepResult

from .settings import load_settings, execution_gate

LABELS = ("x", "y", "z", "pitch", "roll", "gripper")
DOCS = """Physical SO-101 follower: five arm joints and a gripper. move_to targets
are absolute x,y,z in meters in the configured real world frame. pitch and roll
are absolute geometric arm angles in radians, not Euler XYZ or offsets from the
start pose. pitch=pi/2 points the fingers down; roll rotates the jaws about the
approach direction. Yaw follows shoulder pan; arbitrary six-dimensional poses
are unavailable. eef_state is x,y,z,pitch,roll,gripper. Gripper 0=open, 1=closed.
Omitted dimensions retain their measured/current command values. Images are real
top and wrist RGB. Camera poses are unknown unless explicitly calibrated; never
assume the simulated camera directions apply. End-effector pose comes from
encoder forward kinematics, not an external position tracker. motor_positions
uses LeRobot degrees for the five arm motors and 0..100 for the gripper.
measured_gripper_closure is measured opening, while eef_state gripper retains
the commanded closure during a grasp. controller_feedback is [accepted,
position_error_m, orientation_error_rad]. The entire proposed path is checked
through IK, calibrated motor travel, and motor velocity limits before execution.
Commands also check measured tracking and observation freshness. Rejected plans
do not move the arm. Physical objects and the robot cannot pause during LLM
reasoning. Use small motions near objects, maintain clearance, and verify the
achieved pose and grasp in fresh images. Software bounds do not provide collision
avoidance or force sensing. done is a model declaration; real success awaits
an explicit human label. No object positions or success oracle are supplied."""


def hardware_task(scene_config, max_steps=1800):
    path = Path(scene_config).resolve()
    scene = json.loads(path.read_text())
    # Hardware consumes only task identity/instruction/reference image. No MuJoCo
    # objects or simulated success predicates participate in execution or scoring.
    return Task(name=f"so101_hardware_{scene['id']}", scenes=[Scene(
        id=scene["id"], instruction=scene["instruction"], metadata={"replica_scene_config": str(path)})],
        scorer=[], max_steps=max_steps, metadata={"success_scoring": "human_label_pending"})


class SO101HardwareEmbodiment(EmbodimentBase):
    def __init__(self, config_path=None, *, mode="mock", assume_calibration=False,
                 output_dir="outputs/so101", max_steps=1800, policy_seconds=900,
                 auto_start=False, video=True, requires_cartesian_verification=True):
        self.config_path = str(Path(config_path).resolve()) if config_path else str(Path(__file__).with_name("assets")/"mock_config.json")
        self.settings, self.profile = load_settings(self.config_path)
        execution_gate(self.settings, self.profile, mode, assume_calibration, requires_cartesian=requires_cartesian_verification)
        if max_steps < 1 or not math.isfinite(policy_seconds) or policy_seconds <= 0:
            raise ConfigError("Positive hardware step and wall-time limits required")
        self.mode, self.assume_calibration = mode, assume_calibration
        self.output, self.max_steps = Path(output_dir).resolve(), max_steps
        self.policy_seconds, self.auto_start, self.video = policy_seconds, auto_start, video
        self.process = self.worker_log = None
        self.last = None
        self.initial_hash, self.trajectory = None, []
        self._plan_id = None
        self._framework_steps = 0
        self._last_state = None
        self._action_reference = self._approved_reference = None
        self._approval_generation = 0
        self._command_rejection = None
        self.last_feedback = "ready"
        self.worker_path = Path(__file__).with_name("_worker.py")
        self.extra_worker_kwargs = {}
        hz = self.profile["control_hz"]
        xyz_low = [self.profile["workspace_m"][a][0] for a in ("x", "y", "z")]
        xyz_high = [self.profile["workspace_m"][a][1] for a in ("x", "y", "z")]
        gripper_range = abs(self.profile["gripper_closed_motor"]-self.profile["gripper_open_motor"])
        step = (self.profile["linear_speed_m_s"]/hz/math.sqrt(3),)*3
        step += (self.profile["angular_speed_rad_s"]/hz,)*2
        step += (self.profile["gripper_speed_units_s"]/hz/gripper_range,)
        self.docs = DOCS + "\nConfigured world frame: " + self.profile["world_frame_description"]
        self.docs += f"\nDevice mode: {mode}. Kinematics verified in file: {self.profile['kinematics_verified']}; world frame verified in file: {self.profile['world_frame_verified']}; calibration explicitly assumed: {assume_calibration}."
        if mode in ("dry-run", "inspect"):
            self.docs += " Motor writes are disabled. Planned motion does not change the measured robot pose."
        if mode == "mock":
            self.docs += " Devices and camera images are offline test doubles; this is not a real-world evaluation."
        self.info = EmbodimentInfo(name="so101_hardware", is_simulated=mode == "mock", control_hz=hz,
            capabilities=frozenset({"renderable"}), docs=self.docs,
            action_space=Box(shape=(6,), low=np.array(xyz_low+[-math.pi, -math.pi, 0]),
                high=np.array(xyz_high+[math.pi, math.pi, 1]),
                semantics=ActionSemantics(control_mode="eef_abs_pose", rotation_repr="none",
                    gripper="continuous", frame="world", dim_labels=LABELS, max_step=step)),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(name, 256, 256) for name in ("top", "wrist")),
                state=StateSpec((StateField("eef_state", (6,)), StateField("motor_positions", (6,)),
                                 StateField("joint_angles", (5,), "rad"), StateField("measured_gripper_closure", (1,)),
                                 StateField("controller_feedback", (3,))))))

    def _rpc(self, request):
        if self.process is None or self.process.poll() is not None:
            raise EmbodimentFault(f"Hardware worker is not running; see {self.output/'worker.log'}")
        try:
            self.process.stdin.write(json.dumps(request, allow_nan=False)+"\n")
            self.process.stdin.flush()
            if not select.select([self.process.stdout], [], [], max(30., self.policy_seconds+10))[0]:
                (self.output/"STOP").touch()
                raise EmbodimentFault("Hardware worker response timed out; stop requested")
            line = self.process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise EmbodimentFault("Hardware worker connection lost") from exc
        if not line:
            raise EmbodimentFault(f"Hardware worker exited; see {self.output/'worker.log'}")
        reply = json.loads(line)
        if "error" in reply:
            raise EmbodimentFault(f"Hardware worker {reply['error']}: {reply.get('detail', '')}")
        if "state" in reply:
            self.last = {k: v for k, v in reply.items() if k not in ("state", "images_png")}
            self.last["framework_steps"] = self._framework_steps
            self._last_state = reply["state"]
            self.last_feedback = reply["feedback"]["detail"]
        return reply

    def reset(self, scene, *, seed=None):
        self.close()
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output/"STOP").exists():
            raise ConfigError("Hardware output contains STOP; use a new episode directory")
        if self.mode != "mock" and not self.auto_start and not os.isatty(0):
            raise ConfigError("Real hardware scene setup needs an interactive terminal, or explicit --auto-start")
        config = (scene.metadata or {}).get("replica_scene_config")
        scene_data = json.loads(Path(config).read_text()) if config else {}
        reference = scene_data.get("provenance", {}).get("reference_image")
        self.instruction = scene.instruction
        self.initial_hash, self.trajectory, self._plan_id = None, [], None
        self._framework_steps = 0
        self._action_reference = self._approved_reference = None
        self._command_rejection = None
        env = {key: value for key, value in os.environ.items() if not any(word in key.upper() for word in ("API_KEY", "TOKEN", "SECRET"))}
        env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        self.worker_log = (self.output/"worker.log").open("a")
        self.process = subprocess.Popen([self.settings["python"], "-u", str(self.worker_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.worker_log, text=True,
            env=env, start_new_session=True)
        initial = self._rpc(dict(op="open", kwargs=dict(config_path=self.config_path, output=str(self.output),
            mode=self.mode, assume_calibration=self.assume_calibration, max_steps=self.max_steps,
            policy_seconds=self.policy_seconds, reference_image=reference, video=self.video, **self.extra_worker_kwargs)))
        if self.mode != "inspect":
            initial = self._rpc(dict(op="setup"))
        if self.mode != "mock" and not self.auto_start:
            print(f"[SETUP] {self.instruction}\nReference: {reference or 'not provided'}", flush=True)
            answer = input("Arrange the scene; press ENTER to start, or type q to stop: ").strip().lower()
            if answer:
                raise ConfigError("Operator cancelled hardware episode")
        initial = self._rpc(dict(op="start"))
        return self._observation(initial)

    def _observation(self, data):
        state, feedback = data["state"], data["feedback"]
        images = {name: np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"))
                  for name, value in data["images_png"].items()}
        eef = np.array(state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
        self._action_reference = eef.copy()
        motors = np.array([state["motor_positions"][name+".pos"] for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")])
        if self.initial_hash is None:
            digest = hashlib.sha256()
            for name in sorted(images):
                digest.update(images[name].tobytes())
                Image.fromarray(images[name]).save(self.output/f"initial_{name}.png")
            digest.update(eef.tobytes())
            self.initial_hash = digest.hexdigest()
        self.trajectory.append(dict(step=data["steps"], eef=eef.tolist(), motor_positions=motors.tolist(),
                                    device_mode=self.mode, controller_feedback=feedback))
        values = dict(eef_state=eef, motor_positions=motors,
            joint_angles=np.array(list(state["model_joint_angles_rad"].values())),
            measured_gripper_closure=np.array([state["measured_gripper_closure"]]),
            controller_feedback=np.array([float(feedback["accepted"]), feedback["position_error_m"], feedback["orientation_error_rad"]]))
        for name, frame in state["reference_frames"]["cameras"].items():
            values[name+"_camera_calibrated"] = np.array([float(frame["verified"])])
            if frame["verified"]:
                values[name+"_camera_position_world"] = np.array(frame["position_world_m"])
                values[name+"_camera_rotation_world"] = np.array(frame["R_world_from_camera"]).reshape(-1)
                if "K_after_crop_resize" in frame:
                    values[name+"_camera_intrinsics"] = np.array(frame["K_after_crop_resize"]).reshape(-1)
        return Observation(images=images, state=values, instruction=self.instruction,
                           state_time=data["steps"]/self.info.control_hz)

    def preflight(self, targets):
        reply = self._rpc(dict(op="plan", targets=np.asarray(targets).tolist()))
        self._plan_id = reply.get("plan_id") if reply.get("accepted") else None
        self._approved_reference = self._action_reference.copy() if reply.get("accepted") and self._action_reference is not None else None
        if reply.get("accepted"):
            self._approval_generation += 1
        with (self.output/"preflight.jsonl").open("a") as file:
            file.write(json.dumps(reply)+"\n")
        return None if reply.get("accepted") else reply.get("reason", "Hardware preflight rejected the path")

    def approval_context(self):
        """Return the immutable source observation for the currently accepted path."""
        if self._plan_id is None or self._approved_reference is None:
            return None
        return self._approval_generation, self._approved_reference.copy()

    def make_approver(self):
        """Build a limiter that shares the accepted tool path's observation reference."""
        from .approver import HardwarePathApprover
        return HardwarePathApprover(self)

    def step(self, action):
        self._framework_steps += 1
        if action.meta.get("request_stop"):
            data = self._rpc(dict(op="hold"))
            reason = action.meta.get("stop_reason", "policy_stop")
            return StepResult(self._observation(data), terminated=True,
                termination_reason="model_done_unverified" if reason == "done" else reason, info={"success": None})
        data = self._rpc(dict(op="step", target=np.asarray(action.data).tolist(), plan_id=self._plan_id))
        reason = data.get("stopped")
        if not data["feedback"]["accepted"]:
            self._command_rejection = data["feedback"]["detail"]
            reason = self._command_rejection
            if self.last is not None:
                self.last["stopped"] = reason
        return StepResult(self._observation(data), terminated=bool(reason),
            truncated=(data["steps"] if self.mode in ("mock", "execute") else data["planned_control_steps"]) >= self.max_steps,
            termination_reason=reason, info={"success": None, "device_mode": self.mode})

    def release_torque_after_close(self):
        """Explicitly requested by the runner after its operator release prompt."""
        if self.mode!="execute":
            raise ValueError("Torque release is only available for physical execute mode")
        if self.process is not None and self.process.poll() is None:
            raise ValueError("Close the hardware worker before releasing torque")
        config=self.output/"torque_release_config.json"
        config.write_text(json.dumps(self.settings,indent=2))
        output=self.output/"torque_release.json"
        output.unlink(missing_ok=True)
        worker_env={key:value for key,value in os.environ.items()
                    if not any(word in key.upper() for word in ("API_KEY","TOKEN","SECRET"))}
        try:
            with (self.output/"torque_release.log").open("a") as log:
                result=subprocess.run([self.settings["python"],"-u",str(Path(__file__).with_name("_release_torque.py")),
                    "--config",str(config),"--output",str(output)],stdout=subprocess.PIPE,stderr=log,
                    text=True,env=worker_env,timeout=30)
            report=json.loads(output.read_text()) if output.is_file() else dict(status="failed",torque_disabled=None,
                detail=f"Torque helper exited {result.returncode}; see torque_release.log")
        except (OSError,subprocess.TimeoutExpired,ValueError) as exc:
            report=dict(status="failed",torque_disabled=None,error_type=type(exc).__name__,detail=str(exc))
            output.write_text(json.dumps(report,indent=2))
        self.last={**(self.last or {}),"torque_release_status":report["status"],"torque_disabled":report["torque_disabled"]}
        summary=self.output/"hardware_summary.json"
        if summary.is_file():
            data=json.loads(summary.read_text())
            data.update(torque_release_status=report["status"],torque_disabled=report["torque_disabled"])
            summary.write_text(json.dumps(data,indent=2))
        return report

    def close(self):
        if self.process is None:
            return
        try:
            (self.output/"STOP").touch()
            if self.process.poll() is None:
                try:
                    self.process.stdin.write('{"op":"close"}\n')
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=3)
            summary = self.output/"hardware_summary.json"
            if summary.is_file():
                self.last = {**(self.last or {}), **json.loads(summary.read_text())}
                self.last["framework_steps"] = self._framework_steps
                if self._command_rejection:
                    self.last["stopped"] = self._command_rejection
        finally:
            if self.process.stdin:
                self.process.stdin.close()
            if self.process.stdout:
                self.process.stdout.close()
            if self.worker_log:
                self.worker_log.close()
                self.worker_log = None
            self.process = None


def hardware_embodiment(**kwargs):
    return SO101HardwareEmbodiment(**kwargs)

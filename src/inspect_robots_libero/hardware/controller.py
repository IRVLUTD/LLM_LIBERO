"""Adapted from kochIRVL/real_robot/controller.py; source repository stays read-only."""
"""Physical SO101 adapter and bounded Cartesian-to-motor execution."""

import math
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from .kinematics import JOINTS, MOTORS, Kinematics, calibrated_motor_limits, wrap, transform


# A lagging waypoint may take longer than one control tick, but cannot keep
# applying a command indefinitely if an actuator is blocked or not responding.
WAYPOINT_CATCHUP_TIMEOUT_S = 2.0


class StopRequested(RuntimeError):
    pass


class HardwareDriver:
    def __init__(self, *, port, follower_id, calibration_dir, top_index, wrist_index,
                 width=640, height=480, fps=30, execute=False):
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        camera_configs = {
            "top": OpenCVCameraConfig(index_or_path=top_index, width=width, height=height, fps=fps),
            "wrist": OpenCVCameraConfig(index_or_path=wrist_index, width=width, height=height,
                                        fps=fps, fourcc="MJPG", warmup_s=2),
        }
        config = SO101FollowerConfig(port=port, id=follower_id, use_degrees=True,
            calibration_dir=Path(calibration_dir), cameras=camera_configs,
            # Match benchmark.py's LeRobot default. A 2-degree measured-to-goal
            # cap can keep a loaded shoulder from developing enough effort to
            # move. RealController separately bounds command speed/travel and
            # checks arm tracking error before every waypoint.
            max_relative_target=None, disable_torque_on_disconnect=False)
        self.robot = SO101Follower(config)
        self.execute = execute

    def connect(self):
        # Bypass SOFollower.connect/configure: that routine toggles torque and
        # may enter interactive calibration. Inspection/dry-run must not do that.
        bus = self.robot.bus
        try:
            bus.connect()
            if not bus.is_calibrated:
                raise ValueError("Motor EEPROM calibration does not match the selected file; calibrate separately first.")
            from lerobot.motors.feetech.feetech import OperatingMode
            for motor in bus.motors:
                if bus.read("Operating_Mode", motor) != OperatingMode.POSITION.value:
                    raise ValueError(f"{motor} is not in position mode; prepare the robot separately.")
            for camera in self.robot.cameras.values():
                camera.connect()
            motors = self.read_motors()
            if self.execute:
                limits = calibrated_motor_limits({n: vars(c) for n, c in self.robot.calibration.items()})
                values = np.asarray([motors[n] for n in MOTORS])
                if not np.isfinite(values).all() or np.any(values < limits[:, 0] - 0.09) or np.any(values > limits[:, 1] + 0.09):
                    raise ValueError("Measured start pose exceeds calibrated motor travel")
                # Establish a measured-position hold before enabling any disabled
                # motors. Never jump to a reset pose or change calibration here.
                self.send(motors)
                bus.enable_torque()
        except BaseException:
            self.close()
            raise

    def read_motors(self):
        return {n + ".pos": float(v) for n, v in self.robot.bus.sync_read("Present_Position").items()}

    def read_diagnostics(self):
        """Read servo tuning and health without changing goals, torque or EEPROM.

        Values use LeRobot's sign decoding but no calibration normalization.
        Keep register units explicit rather than guessing firmware conversions.
        Individual unsupported/failed reads remain visible in the report.
        """
        registers = (
            "Operating_Mode", "Torque_Enable", "P_Coefficient", "I_Coefficient", "D_Coefficient",
            "Max_Torque_Limit", "Torque_Limit", "Protection_Current", "Overload_Torque",
            "Acceleration", "Goal_Time", "Goal_Velocity", "Goal_Position", "Present_Position",
            "Present_Velocity", "Present_Load", "Present_Current", "Present_Voltage",
            "Present_Temperature", "Status", "Moving",
        )
        motors = {}
        for name, motor in self.robot.bus.motors.items():
            values, errors = {}, {}
            for register in registers:
                try:
                    values[register] = self.robot.bus.read(register, name, normalize=False)
                except Exception as exc:
                    errors[register] = f"{type(exc).__name__}: {exc}"
            motors[name] = dict(id=motor.id, model=motor.model, registers=values, read_errors=errors)
        return dict(source="physical_servo_registers", units="Register values with sign decoding; no calibration normalization",
                    writes_performed=0, motors=motors)

    def read_images(self):
        # The installed OpenCV driver rejects frames older than 500 ms.
        return {name: camera.read_latest().copy() for name, camera in self.robot.cameras.items()}

    def send(self, motors):
        if not self.execute:
            raise RuntimeError("Motor writes are disabled in read-only/dry-run mode")
        return self.robot.send_action(motors)

    def close(self):
        for camera in self.robot.cameras.values():
            if camera.is_connected:
                try:
                    camera.disconnect()
                except Exception:
                    pass
        if self.robot.bus.is_connected:
            # Keep the established hold. Do not drop a carried object by
            # implicitly removing torque when a model/API/session stops.
            self.robot.bus.disconnect(disable_torque=False)


class MockDriver:
    """Offline test double, clearly labeled; not a manipulation simulation."""
    def __init__(self,state_file=None):
        self.motors = dict(zip(MOTORS, [0., -30., 40., 30., 0., 65.]))
        self.sent = []
        self.state_file=Path(state_file) if state_file else None
        if self.state_file is not None and self.state_file.is_file():
            self.motors=json.loads(self.state_file.read_text())

    def connect(self):
        pass

    def read_motors(self):
        return dict(self.motors)

    def read_diagnostics(self):
        return dict(source="mock_device", writes_performed=0, motors={},
                    note="No physical servo registers were read")

    def read_images(self):
        images = {}
        for name in ("top", "wrist"):
            image = Image.new("RGB", (640, 480), (35, 35, 35))
            draw = ImageDraw.Draw(image)
            draw.text((30, 30), f"MOCK DEVICE - {name} - NO REAL SCENE", fill="white")
            draw.text((30, 60), "Offline controller/bridge test only", fill="yellow")
            images[name] = np.asarray(image).copy()
        return images

    def send(self, motors):
        self.sent.append(dict(motors))
        self.motors = dict(motors)
        if self.state_file is not None:
            self.state_file.parent.mkdir(parents=True,exist_ok=True)
            temp=self.state_file.with_suffix(".json.tmp")
            temp.write_text(json.dumps(self.motors))
            temp.replace(self.state_file)
        return dict(motors)

    def close(self):
        pass


class RealController:
    def __init__(self, driver, profile, calibration, *, execute=False, mock=False, zoom=1.5,
                 max_steps=1800, stop_file=None, on_tick=None):
        self.driver, self.profile = driver, profile
        self.kinematics = Kinematics(profile)
        self.motor_limits = calibrated_motor_limits(calibration)
        self.execute, self.mock = execute, mock
        self.zoom = zoom
        self.max_steps = max_steps
        self.steps = 0
        self.stop_file = Path(stop_file) if stop_file else None
        self.on_tick = on_tick
        self.dt = 1 / profile["control_hz"]
        self.observed_at = None
        self.observed_motors = None
        self.gripper_command = None
        self.last_gripper_motor_goal = None
        self.last_motor_goal = None
        self.last_plan = None
        self.deadline = float("inf")
        self.last_tick = 0.0
        self.on_motor_action = None

    def check_stop(self):
        if self.stop_file and self.stop_file.exists():
            raise StopRequested("STOP file requested a stop")
        if time.monotonic() >= self.deadline:
            raise StopRequested("episode wall-time limit")

    def pump(self):
        self.check_stop()
        if self.on_tick and time.monotonic() - self.last_tick >= 0.1:
            self.last_tick = time.monotonic()
            self.on_tick(self.images())

    def valid_motors(self, motors):
        if set(motors) != set(MOTORS):
            raise ValueError("Every motor target/reading must contain exactly the six .pos keys")
        values = np.asarray([motors[n] for n in MOTORS], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite motor reading/target")
        # One encoder-count tolerance allows integer truncation in the bus conversion.
        if np.any(values < self.motor_limits[:, 0] - 0.09) or np.any(values > self.motor_limits[:, 1] + 0.09):
            raise ValueError("Motor target/reading exceeds the selected calibration travel")
        return values

    def images(self):
        import cv2
        result = {}
        for name, raw in self.driver.read_images().items():
            height, width = raw.shape[:2]
            cw, ch = max(1, int(width / self.zoom)), max(1, int(height / self.zoom))
            left, top = (width - cw) // 2, (height - ch) // 2
            image = cv2.resize(raw[top:top + ch, left:left + cw], (width, height))
            rendered = Image.fromarray(image)
            draw = ImageDraw.Draw(rendered)
            calibration = self.profile.get("camera_calibration", {}).get(name, {})
            label = f"{name} | {'camera pose calibrated' if calibration.get('verified') else 'camera pose UNCALIBRATED'}"
            draw.rectangle((3, 3, min(width - 3, 420), 23), fill=(15, 15, 15))
            draw.text((7, 7), label, fill="white")
            result[name] = np.asarray(rendered).copy()
        return result

    def camera_frames(self, pose, images):
        frames = {}
        for name, image in images.items():
            cfg = self.profile.get("camera_calibration", {}).get(name, {})
            frame = {"verified": bool(cfg.get("verified")), "image_size_px": [image.shape[1], image.shape[0]]}
            if not frame["verified"]:
                frame["note"] = "Camera-to-robot pose is unknown. Do not infer world directions from the simulation cameras."
                frames[name] = frame
                continue
            key = "world_from_camera" if name == "top" else "tool_from_camera"
            matrix = np.asarray(cfg[key], dtype=float)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid camera transform for {name}")
            if not np.allclose(matrix[3], [0, 0, 0, 1]):
                raise ValueError(f"Invalid homogeneous camera transform for {name}")
            rotation = matrix[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.99:
                raise ValueError(f"Camera rotation for {name} is not a proper rotation")
            if name == "wrist":
                matrix = pose @ matrix
            frame.update(position_world_m=matrix[:3, 3].tolist(), R_world_from_camera=matrix[:3, :3].tolist(),
                convention="OpenCV: +X image right, +Y image down, +Z optical forward")
            if "K" in cfg:
                k = np.asarray(cfg["K"], dtype=float).copy()
                width, height = image.shape[1], image.shape[0]
                if cfg.get("image_size_px") != [width, height] or k.shape != (3, 3) or not np.isfinite(k).all():
                    raise ValueError(f"Camera intrinsics/resolution mismatch for {name}")
                if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
                    raise ValueError(f"Invalid camera intrinsics for {name}")
                cw, ch = max(1, int(width / self.zoom)), max(1, int(height / self.zoom))
                k[0, 2] -= (width - cw) // 2
                k[1, 2] -= (height - ch) // 2
                k[0] *= width / cw
                k[1] *= height / ch
                frame["K_after_crop_resize"] = k.tolist()
                frame["distortion_coefficients"] = cfg.get("distortion_coefficients", [])
            frames[name] = frame
        return frames

    def observe(self):
        self.check_stop()
        motors = self.driver.read_motors()
        self.valid_motors(motors)
        q = self.kinematics.motor_to_model(motors)
        pose = self.kinematics.forward(q)
        images = self.images()
        pitch, roll = self.kinematics.orientation(pose)
        self.observed_at = time.monotonic()
        self.observed_motors = dict(motors)
        opened, closed = self.profile["gripper_open_motor"], self.profile["gripper_closed_motor"]
        measured_closure = float(np.clip((motors["gripper.pos"] - opened) / (closed - opened), 0, 1))
        if self.gripper_command is None:
            self.gripper_command = measured_closure
        return {
            "mode": "mock" if self.mock else ("execute" if self.execute else "dry_run"),
            "motor_positions": motors, "arm_motor_units": "LeRobot degrees", "gripper_motor_units": "LeRobot 0..100",
            "model_joint_angles_rad": dict(zip(JOINTS, q.tolist())),
            "outside_model_joint_limits": [n for i, n in enumerate(JOINTS) if not self.kinematics.limits[i, 0] <= q[i] <= self.kinematics.limits[i, 1]],
            "eef_position_m": pose[:3, 3].tolist(), "eef_rotation_matrix": pose[:3, :3].tolist(),
            "eef_position_source": "forward kinematics from encoders; not externally measured",
            "pitch_rad": float(pitch), "roll_rad": float(roll),
            "gripper_command": self.gripper_command, "measured_gripper_closure": measured_closure,
            "control_steps": self.steps,
            "reference_frames": {"action_frame": "configured world", "description": self.profile["world_frame_description"],
                "kinematics_verified": self.profile["kinematics_verified"], "world_frame_verified": self.profile["world_frame_verified"],
                "calibration_assumed": bool(self.profile.get("calibration_assumed", False)),
                "calibration_assumption_details": self.profile.get("calibration_assumption"),
                "world_from_base": self.kinematics.world_from_base.tolist(),
                "R_world_from_tool": pose[:3, :3].tolist(), "approach_world": (-pose[:3, 1]).tolist(),
                "cameras": self.camera_frames(pose, images)},
        }, images

    def hold(self):
        if self.execute:
            motors = self.driver.read_motors()
            if self.last_gripper_motor_goal is not None:
                # Keep the existing grasp command rather than relaxing the
                # fingers to their measured contact position during a stop.
                motors["gripper.pos"] = self.last_gripper_motor_goal
            self.valid_motors(motors)
            self.last_motor_goal = dict(self.driver.send(motors))

    def execute_path(self, path):
        self.last_plan = path
        if not self.execute:
            return {"status": "planned_only", "planned_control_steps": len(path),
                    "first_motor_target": path[0], "last_motor_target": path[-1]}
        if self.steps + len(path) > self.max_steps:
            return {"status": "rejected", "reason": "control-step budget would be exceeded"}
        first_step = self.steps
        try:
            for target in path:
                catchup_deadline = None
                clamped = {}
                waiting_motor = None
                while True:
                    started = time.monotonic()
                    self.check_stop()
                    if catchup_deadline is not None and started >= catchup_deadline:
                        self.hold()
                        if waiting_motor is not None:
                            return {"status": "tracking_failed",
                                    "reason": "Motor did not catch up within 2 seconds; next waypoint would exceed the tracking limit",
                                    **waiting_motor}
                        return {"status": "tracking_failed",
                                "reason": "Motor did not catch up to the clamped waypoint within 2 seconds",
                                "clamped_motors": clamped}
                    if self.steps >= self.max_steps:
                        self.hold()
                        return {"status": "tracking_failed", "reason": "control-step budget exhausted during motor catch-up"}
                    self.valid_motors(target)
                    current = self.driver.read_motors()
                    self.valid_motors(current)
                    # Tracking refers to the command already sent, not the next
                    # waypoint. Pause path advancement before a new goal would
                    # exceed the same limit, allowing the latched goal to finish.
                    previous = self.last_motor_goal or current
                    motor = max(MOTORS[:5], key=lambda n: abs(previous[n] - current[n]))
                    error = abs(previous[motor] - current[motor])
                    if error > self.profile["max_tracking_error_deg"]:
                        self.hold()
                        return {"status": "tracking_failed", "reason": "Motor tracking error exceeded configured degrees",
                                "motor": motor, "error_deg": error,
                                "requested_position": previous[motor], "measured_position": current[motor]}
                    motor = max(MOTORS[:5], key=lambda n: abs(target[n] - current[n]))
                    error = abs(target[motor] - current[motor])
                    if error > self.profile["max_tracking_error_deg"]:
                        waiting_motor = {"motor": motor, "error_deg": error,
                                         "requested_position": target[motor], "measured_position": current[motor],
                                         "last_sent_position": previous[motor]}
                        if catchup_deadline is None:
                            catchup_deadline = started + WAYPOINT_CATCHUP_TIMEOUT_S
                        # No new goal and no measured-position hold here: either
                        # would disrupt the actuator's progress toward its goal.
                        self.pump()
                        if not self.mock:
                            time.sleep(max(0, self.dt - (time.monotonic() - started)))
                        continue
                    waiting_motor = None
                    sent = self.driver.send(target)
                    self.last_motor_goal = dict(sent)
                    self.last_gripper_motor_goal = sent["gripper.pos"]
                    if self.on_motor_action:
                        self.on_motor_action(target, sent, current)
                    self.steps += 1
                    clamped = {n: {"requested": target[n], "sent": sent[n], "measured_before": current[n]}
                               for n in MOTORS if abs(target[n] - sent[n]) > 1e-4}
                    self.pump()
                    if not self.mock:
                        time.sleep(max(0, self.dt - (time.monotonic() - started)))
                    if not clamped:
                        break
                    # Retry this SAME waypoint at the normal control rate. The
                    # driver's relative-position cap still applies on every send;
                    # the planned path cannot run ahead while the motor catches up.
                    if catchup_deadline is None:
                        catchup_deadline = started + WAYPOINT_CATCHUP_TIMEOUT_S
        except BaseException:
            try:
                self.hold()
            except Exception:
                pass
            raise
        return {"status": "executed", "control_steps": self.steps - first_step}

    def reset(self, target):
        """Bounded joint-space reset; honors motor calibration, not IK URDF limits."""
        end = self.valid_motors(target)
        current = self.driver.read_motors()
        start = self.valid_motors(current)
        speeds = np.array([self.profile["joint_speed_deg_s"]] * 5 + [self.profile["gripper_speed_units_s"]])
        count = max(1, math.ceil(float(np.max(np.abs(end - start) / speeds)) / self.dt))
        path = [dict(zip(MOTORS, (start + t * (end - start)).tolist())) for t in np.linspace(1 / count, 1, count)]
        return self.execute_path(path)

    def move_to(self, targets):
        self.last_plan = None
        if not isinstance(targets, dict) or not targets:
            return {"status": "rejected", "reason": "targets must be a nonempty object"}
        bounds = {**self.profile["workspace_m"], "pitch": [-math.pi, math.pi],
                  "roll": [-math.pi, math.pi], "gripper": [0, 1]}
        for name, value in targets.items():
            if name not in bounds or isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                return {"status": "rejected", "reason": f"Unknown or non-finite dimension: {name}"}
            if not bounds[name][0] <= value <= bounds[name][1]:
                return {"status": "rejected", "reason": f"{name} outside {bounds[name]}"}
        if self.observed_at is None or time.monotonic() - self.observed_at > self.profile["observation_max_age_s"]:
            return {"status": "rejected", "reason": "Observation is stale; inspect the refreshed images before another action"}
        current = self.driver.read_motors()
        self.valid_motors(current)
        if max(abs(current[n] - self.observed_motors[n]) for n in MOTORS[:5]) > 3:
            return {"status": "rejected", "reason": "Arm moved since the observation; inspect the refreshed state"}
        q = self.kinematics.motor_to_model(current)
        start_pose = self.kinematics.forward(q)
        start = start_pose[:3, 3]
        start_angles = self.kinematics.orientation(start_pose)
        destination = np.array([targets.get(n, start[i]) for i, n in enumerate(("x", "y", "z"))])
        angles = np.array([targets.get(n, start_angles[i]) for i, n in enumerate(("pitch", "roll"))])
        required = np.array([n in targets for n in ("pitch", "roll")])
        delta_angles = wrap(angles - start_angles)
        distance = float(np.linalg.norm(destination - start))
        if distance > self.profile["max_translation_m"]:
            return {"status": "rejected", "reason": f"Translation exceeds {self.profile['max_translation_m']} m per call"}
        gripper = float(targets.get("gripper", self.gripper_command))
        end_gripper = self.profile["gripper_open_motor"] + gripper * (
            self.profile["gripper_closed_motor"] - self.profile["gripper_open_motor"])
        duration = max(distance / self.profile["linear_speed_m_s"],
            float(np.max(np.abs(delta_angles))) / self.profile["angular_speed_rad_s"],
            abs(end_gripper - current["gripper.pos"]) / self.profile["gripper_speed_units_s"], self.dt)
        count = math.ceil(duration / self.dt)
        has_pose_target = any(n in targets for n in ("x", "y", "z", "pitch", "roll"))
        if has_pose_target and (np.any(q < self.kinematics.limits[:, 0]) or np.any(q > self.kinematics.limits[:, 1])):
            return {"status": "rejected", "reason": "Start pose is outside nominal model joint limits. Inspect the model mapping/verified limit overrides or choose an in-range reset pose."}
        for attempt in range(4):
            if count + 10 > self.max_steps - self.steps:
                return {"status": "rejected", "reason": "Insufficient control steps for a bounded trajectory"}
            path, seed = [], q.copy()
            for fraction in np.linspace(1 / count, 1, count):
                if has_pose_target:
                    waypoint = start + fraction * (destination - start)
                    for i, name in enumerate(("x", "y", "z")):
                        if not bounds[name][0] <= waypoint[i] <= bounds[name][1]:
                            return {"status": "rejected", "reason": "Cartesian path leaves configured workspace"}
                    seed, pe, re = self.kinematics.solve(waypoint, start_angles + fraction * delta_angles,
                                                       seed, required, self.motor_limits)
                    if pe > self.profile["position_tolerance_m"] or re > 0.1:
                        return {"status": "rejected", "reason": "No feasible IK path", "ik_position_error_m": pe,
                                "ik_orientation_error_rad": re, "path_fraction": float(fraction)}
                motors = self.kinematics.model_to_motor(seed, current["gripper.pos"] + fraction * (end_gripper - current["gripper.pos"]))
                self.valid_motors(motors)
                path.append(motors)
            vectors = np.array([[current[n] for n in MOTORS]] + [[p[n] for n in MOTORS] for p in path])
            speeds = np.array([self.profile["joint_speed_deg_s"]] * 5 + [self.profile["gripper_speed_units_s"]])
            ratio = float(np.max(np.abs(np.diff(vectors, axis=0)) / (speeds * self.dt)))
            if ratio <= 1.001:
                break
            count = math.ceil(count * ratio * 1.05)
        else:
            return {"status": "rejected", "reason": "Could not satisfy motor velocity limits"}
        # Check the real arm again after IK preflight, before any motor write.
        fresh = self.driver.read_motors()
        self.valid_motors(fresh)
        if time.monotonic() - self.observed_at > self.profile["observation_max_age_s"]:
            return {"status": "rejected", "reason": "Observation expired during planning; inspect refreshed images"}
        if max(abs(fresh[n] - current[n]) for n in MOTORS[:5]) > 2:
            return {"status": "rejected", "reason": "Arm moved during planning; reobserve"}
        result = self.execute_path(path)
        if result["status"] == "executed":
            self.gripper_command = gripper
            for _ in range(10):
                self.pump()
                if not self.mock:
                    time.sleep(self.dt)
            measured = self.driver.read_motors()
            pose = self.kinematics.forward(self.kinematics.motor_to_model(measured))
            error = float(np.linalg.norm(pose[:3, 3] - destination))
            orientation_error = float(np.max(np.abs(wrap(self.kinematics.orientation(pose) - angles)) * required))
            result.update(status="reached" if error <= self.profile["position_tolerance_m"] * 1.5 and orientation_error <= 0.15 else "partial",
                          position_error_m=error, orientation_error_rad=orientation_error, achieved_motor_positions=measured)
        return result

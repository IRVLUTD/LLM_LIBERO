"""Real upstream SOArmEmbodiment for both joint and Cartesian benchmark variants.

The injected LeRobot I/O facade preserves the existing isolated-worker serial,
camera and hold lifecycle. Every reset/task waypoint executes upstream.step().
The Cartesian variant retains our IK planner; the joint variant never calls IK.
"""
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from .runtime import HardwareRuntime
from .controller import StopRequested
from .kinematics import MOTORS
from .upstream_support import upstream_config, upstream_provenance
from .settings import load_settings


class UpstreamRuntime(HardwareRuntime):
    """Share device logging and guards while executing the genuine upstream adapter."""
    def __init__(self, *args, control="joint", **kwargs):
        if control not in ("joint", "cartesian"):
            raise ValueError("Upstream control must be joint or cartesian")
        provenance = upstream_provenance()
        config_path = args[0] if args else kwargs.get("config_path")
        settings, profile = load_settings(config_path)
        body_config = upstream_config(settings, profile)
        self.control, self.body, self.raw_send = control, None, None
        self.upstream_steps = 0
        self.confirmed_task_steps = self.confirmed_reset_steps = 0
        self.latest_upstream_info = {}
        super().__init__(*args, requires_cartesian_verification=control == "cartesian", **kwargs)
        try:
            from inspect_robots_so101.embodiment import SOArmEmbodiment
            from inspect_robots_so101.operator import OperatorIO
            runtime = self
            self.raw_send = self.driver.send

            class DeviceFacade:
                """Public upstream driver seam backed by our existing LeRobot device I/O."""
                def get_observation(self):
                    runtime.controller.check_stop()
                    return {**runtime.driver.read_motors(), **runtime.controller.images()}

                def send_action(self, action):
                    runtime.controller.check_stop()
                    runtime.controller.valid_motors(action)
                    if not runtime.controller.execute:
                        raise RuntimeError("Read-only upstream driver cannot write motors")
                    accepted = runtime.raw_send(dict(action))
                    if runtime.started:
                        runtime.confirmed_task_steps += 1
                    else:
                        runtime.confirmed_reset_steps += 1
                    return accepted

                def disconnect(self):
                    # HardwareRuntime owns this connected device and its final hold.
                    pass

            self.body = SOArmEmbodiment(config=body_config,
                driver_factory=lambda config: DeviceFacade(),
                operator=OperatorIO(input_fn=lambda prompt: "", output_fn=lambda message: None),
                poll_end=lambda: False,
                **({"sleep_fn": lambda seconds: None} if self.mode == "mock" else {}))
            self.body.reset(Scene(id="upstream_worker", instruction="Operator setup is handled by the parent benchmark"))

            def via_upstream(motors):
                if self._holding:
                    return self.raw_send(motors)
                self.upstream_steps += 1
                self.event("upstream_step_started", step=self.upstream_steps, control_interface=self.control)
                result = self.body.step(Action(data=np.array([motors[key] for key in MOTORS])))
                self.latest_upstream_info = dict(result.info)
                self.event("upstream_step", step=self.upstream_steps, control_interface=self.control,
                           adapter=provenance["embodiment_class"], info=self.latest_upstream_info)
                # A timeout is observable upstream, but the physical benchmark must
                # stop rather than silently proceed from an unconverged arm.
                if result.info.get("settled") is False or result.info.get("settle_disabled"):
                    raise StopRequested("Upstream arm settling failed; further motor commands blocked")
                return dict(self.controller.last_motor_goal)

            self.driver.send = via_upstream
            snapshot_path = self.output/"hardware_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text())
            snapshot.update(upstream=provenance, control_interface=control,
                driver_seam="Existing isolated LeRobot I/O injected through SOArmEmbodiment.driver_factory",
                homing="Existing bounded startup reset routed through upstream.step; upstream home_pose unset",
                requires_cartesian_calibration=control == "cartesian")
            snapshot_path.write_text(json.dumps(snapshot, indent=2))
        except BaseException:
            self.close()
            raise

    def setup(self):
        """Keep the known startup pose, with each bounded reset packet using upstream.step."""
        observed = super().setup()
        self.upstream_steps = 0
        self.body.num_steps = 0
        return observed

    def plan(self, targets):
        """Joint baseline validates joint paths; Cartesian variant uses the existing IK planner."""
        if self.control == "cartesian":
            return super().plan(targets)
        self.approved.clear()
        ctrl = self.controller
        ctrl.check_stop()
        array = np.asarray(targets, dtype=float)
        if array.ndim != 2 or array.shape[1] != 6 or not 1 <= len(array) <= self._remaining() or not np.isfinite(array).all():
            return dict(accepted=False, reason="Expected a finite native N-by-6 joint path within the remaining step budget")
        if ctrl.observed_at is None or time.monotonic()-ctrl.observed_at > self.profile["observation_max_age_s"]:
            return dict(accepted=False, reason="Joint observation expired")
        current = self.driver.read_motors()
        ctrl.valid_motors(current)
        if max(abs(current[n]-ctrl.observed_motors[n]) for n in MOTORS[:5]) > 3:
            return dict(accepted=False, reason="Arm drifted since the joint observation")
        previous = np.array([(ctrl.last_motor_goal or current)[n] for n in MOTORS])
        limits = np.array([self.profile["joint_speed_deg_s"]]*5+[self.profile["gripper_speed_units_s"]])*ctrl.dt
        retained_gripper = ctrl.last_gripper_motor_goal
        hold_gripper = retained_gripper is not None and abs(array[-1, 5]-retained_gripper) < 1e-7
        segments, planned = [], 0
        for target in array:
            ctrl.valid_motors(dict(zip(MOTORS, target.tolist())))
            command = target.copy()
            if hold_gripper:
                command[5] = retained_gripper
            count = max(1, int(np.ceil(np.max(np.abs(command-previous)/limits))))
            planned += count
            if planned > self._remaining():
                return dict(accepted=False, reason="Native joint motor interpolation exceeds remaining step budget")
            path = [dict(zip(MOTORS, (previous+fraction*(command-previous)).tolist()))
                    for fraction in np.linspace(1/count, 1, count)]
            segments.append(dict(target=target.tolist(), path=path))
            previous = command
        self.plan_id += 1
        self.first_waypoint = True
        self.plan_start_motors = dict(current)
        self.approved.extend(segments)
        self.event("native_preflight", plan_id=self.plan_id, targets=array.tolist(), motor_steps=planned,
                   retained_gripper=retained_gripper if hold_gripper else None, adapter="inspect-robots-so101")
        return dict(accepted=True, plan_id=self.plan_id, motor_steps=planned)

    def step(self, target, plan_id):
        """Execute an approved joint target directly, or an IK-planned Cartesian segment."""
        if self.control == "cartesian":
            return super().step(target, plan_id)
        ctrl = self.controller
        ctrl.check_stop()
        target = np.asarray(target, dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all() or not self.approved or plan_id != self.plan_id or not np.allclose(target, self.approved[0]["target"], rtol=0, atol=1e-7):
            self.approved.clear()
            self.feedback = dict(accepted=False, position_error_m=0., orientation_error_rad=0., detail="No matching native joint preflight")
            return self.capture_observation()
        if self.first_waypoint:
            fresh = self.driver.read_motors()
            ctrl.valid_motors(fresh)
            if time.monotonic()-ctrl.observed_at > self.profile["observation_max_age_s"] or max(abs(fresh[n]-self.plan_start_motors[n]) for n in MOTORS[:5]) > 2:
                self.approved.clear()
                self.feedback = dict(accepted=False, position_error_m=0., orientation_error_rad=0., detail="Native observation expired or arm drifted after preflight")
                return self.capture_observation()
            self.first_waypoint = False
        segment = self.approved.popleft()
        self.planned_steps += len(segment["path"])
        result = ctrl.execute_path(segment["path"])
        self.framework_steps += 1
        if result["status"] == "executed":
            ctrl.gripper_command = float(np.clip((ctrl.last_gripper_motor_goal-self.profile["gripper_open_motor"])/(self.profile["gripper_closed_motor"]-self.profile["gripper_open_motor"]), 0, 1))
        if result["status"] not in ("executed", "planned_only"):
            self.approved.clear()
            self.stopped = result.get("reason", result["status"])
            if result["status"] != "tracking_failed":
                self.hold()
        self.feedback = dict(accepted=result["status"] in ("executed", "planned_only"), position_error_m=0., orientation_error_rad=0., detail=result["status"])
        return self.capture_observation()

    def summary(self):
        """Report the selected action contract and actual upstream step invocations."""
        return dict(super().summary(), control_interface=self.control, upstream_steps=self.upstream_steps,
                    steps=self.confirmed_task_steps, reset_control_steps=self.confirmed_reset_steps,
                    upstream_step_info=self.latest_upstream_info)

    def close(self):
        """Release the upstream facade, then preserve the established physical hold."""
        try:
            if self.body is not None:
                self.body.close()
                self.body = None
        finally:
            if self.raw_send is not None and self.driver is not None:
                self.driver.send = self.raw_send
            super().close()

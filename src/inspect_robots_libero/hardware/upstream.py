"""Separate benchmark adapters using the pinned upstream SOArmEmbodiment."""
import base64
from dataclasses import replace
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
from inspect_robots.spaces import StateField, StateSpec
from inspect_robots.types import Observation

from .embodiment import SO101HardwareEmbodiment
from .upstream_support import upstream_config, upstream_provenance

JOINT_DOCS = """SO-101 controlled through the upstream inspect-robots-so101
SOArmEmbodiment joint-position interface. Use move_joints with absolute native
targets: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll are
LeRobot degrees; gripper is the LeRobot native 0..100 motor position. This is
joint control. There is no Cartesian target tool and no IK in the native policy
path. joint_pos contains measured encoder positions in that order. The gripper
opening directions are given below. A finger blocked by an object can differ
from its commanded position; gripper_target_native records that target. For a
maintained grasp, explicitly retain that gripper target in each move_joints
request. Current and previous top/wrist RGB images show the physical scene.
Use small deliberate joint changes and confirm the achieved state from fresh
observations. Calibrated travel, motor speed, observation age and tracking are
checked. Native joint control does not check Cartesian workspace or plan around
collisions. done is unverified until a human labels success."""


class SO101UpstreamEmbodiment(SO101HardwareEmbodiment):
    """Choose native joint control or a Cartesian bridge over genuine upstream steps."""
    def __init__(self, *args, control="joint", **kwargs):
        if control not in ("joint", "cartesian"):
            raise ValueError("control must be joint or cartesian")
        self.control = control
        self.upstream = upstream_provenance()
        super().__init__(*args, requires_cartesian_verification=control == "cartesian", **kwargs)
        self.worker_path = Path(__file__).with_name("_upstream_worker.py")
        self.extra_worker_kwargs = {"control": control}
        self.upstream_settings = upstream_config(self.settings, self.profile)
        if control == "joint":
            from inspect_robots_so101.embodiment import SOArmEmbodiment
            from inspect_robots_so101.packing import MOTORS
            native = SOArmEmbodiment(config=self.upstream_settings).info
            speed = tuple([self.profile["joint_speed_deg_s"]/native.control_hz]*5 +
                          [self.profile["gripper_speed_units_s"]/native.control_hz])
            semantics = replace(native.action_space.semantics, dim_labels=MOTORS, max_step=speed)
            self.docs = JOINT_DOCS + f"\nGripper open={self.profile['gripper_open_motor']}; closed={self.profile['gripper_closed_motor']}. Device mode={self.mode}."
            if self.mode == "mock":
                self.docs += " Devices/images are offline test doubles, not a real manipulation scene."
            if self.mode in ("inspect", "dry-run"):
                self.docs += " Motor writes are disabled; planned targets do not change measured pose."
            obs_space = replace(native.observation_space,
                cameras=self.info.observation_space.cameras,
                state_keys=frozenset({"joint_pos", "gripper_target_native", "command_accepted"}),
                state=StateSpec((StateField("joint_pos", (6,), "deg+normalized"),
                                 StateField("gripper_target_native", (1,)), StateField("command_accepted", (1,)))))
            self.info = replace(native, name="so101_upstream_joint", is_simulated=self.mode == "mock",
                docs=self.docs, observation_space=obs_space,
                action_space=replace(native.action_space, semantics=semantics))
        else:
            self.docs += "\nThe Cartesian IK/planner is provided by LLM_LIBERO; each planned motor waypoint is executed by the pinned upstream SOArmEmbodiment.step, including its joint clamp and pacing."
            self.info = replace(self.info, name="so101_upstream_cartesian", docs=self.docs)

    def _observation(self, data):
        if self.control == "cartesian":
            return super()._observation(data)
        from inspect_robots_so101.packing import MOTORS
        state = data["state"]
        images = {name: np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"))
                  for name, value in data["images_png"].items()}
        joints = np.array([state["motor_positions"][name+".pos"] for name in MOTORS])
        self._action_reference = joints.copy()
        gripper = self.profile["gripper_open_motor"] + state["gripper_command"]*(self.profile["gripper_closed_motor"]-self.profile["gripper_open_motor"])
        if self.initial_hash is None:
            digest = hashlib.sha256()
            for name in sorted(images):
                digest.update(images[name].tobytes())
                Image.fromarray(images[name]).save(self.output/f"initial_{name}.png")
            digest.update(joints.tobytes())
            self.initial_hash = digest.hexdigest()
        self.trajectory.append(dict(step=data["steps"], joint_pos=joints.tolist(),
            eef=state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]],
            control_interface="joint", upstream_steps=data.get("upstream_steps")))
        return Observation(images=images, state={"joint_pos": joints,
            "gripper_target_native": np.array([gripper]),
            "command_accepted": np.array([float(data["feedback"]["accepted"])])},
            instruction=self.instruction, state_time=data["steps"]/self.info.control_hz)


def upstream_joint_embodiment(**kwargs):
    """Registry factory for the native joint-contract benchmark."""
    return SO101UpstreamEmbodiment(control="joint", **kwargs)


def upstream_cartesian_embodiment(**kwargs):
    """Registry factory for the Cartesian-to-upstream benchmark."""
    return SO101UpstreamEmbodiment(control="cartesian", **kwargs)

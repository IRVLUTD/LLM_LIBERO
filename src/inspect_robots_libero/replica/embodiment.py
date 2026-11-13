"""SO-101 adapter for the same Inspect Robots policy used by LIBERO."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
from inspect_robots.embodiment import EmbodimentBase, EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec
from inspect_robots.task import Task
from inspect_robots.types import Observation, StepResult

from .scenes import read_scene, sha256

LABELS = ("x", "y", "z", "pitch", "roll", "gripper")
LOW = np.array([.04, -.24, .008, -math.pi, -math.pi, 0.])
HIGH = np.array([.42, .24, .40, math.pi, math.pi, 1.])
DOCS = """Single SO-101 arm, five arm joints plus gripper. move_to targets use fixed
world-frame x,y,z in meters (table z=0; +X into workspace, +Z up). The angle labels
pitch and roll are this arm's absolute geometric angles in radians, not Euler
XYZ or angles relative to the starting pose. pitch=pi/2 points the fingers down;
roll rotates the jaws about the approach axis. Shoulder pan determines yaw;
arbitrary six-dimensional poses cannot be commanded. eef_state lists x,y,z,pitch,
roll,gripper in that order. Gripper 0=open, 1=closed; intermediate openings are
supported. Gripper command does not prove a grasp: check measured finger angle,
pad separation, both images, and the next observation after lifting. Unnamed
dimensions retain their measured values. The tool site is on the fixed jaw near
its tip, not the jaw midpoint. Motions are interpolated at 20 Hz, and IK drives
position actuators through physics. Always inspect achieved pose and controller
feedback. Unreachable waypoints hold the arm and consume a control tick. Lift
to safe clearance before lateral travel; use small corrections near objects.
The top and wrist cameras show only RGB and a projected world-axis compass.
Proprioception contains robot state only; no task-object positions or private
score are supplied. Verify the task from images before declaring done.
controller_feedback contains [accepted, IK_position_error_m, IK_angle_error_rad];
accepted=1 means applied, 0 means rejected. Errors are zero when no IK was needed."""


def replica_task(scene_config: str, max_steps: int = 1800) -> Task:
    config = Path(scene_config).resolve()
    scene = read_scene(config)
    if max_steps < 1:
        raise ConfigError("max_steps must be positive")
    return Task(name=f"replica_{scene['id']}", scenes=[Scene(
        id=scene["id"], instruction=scene["instruction"], init_seed=scene["seed"],
        metadata={"replica_scene_config": str(config)})],
        scorer=success_at_end(), max_steps=max_steps)


class ReplicaEmbodiment(EmbodimentBase):
    def __init__(self, scene_config: str | None = None, output_dir="outputs/replica",
                 max_steps=1800, video=True, viewer=False):
        self.scene_config = Path(scene_config).resolve() if scene_config else None
        self.output, self.max_steps, self.video_enabled = Path(output_dir), max_steps, video
        self.sim = self.scorer = self.video = None
        self.viewer_enabled, self.viewer = viewer, None
        self.last = None
        self.initial_hash = None
        self.trajectory = []
        self.feedback = "ready"
        self.info = EmbodimentInfo(
            name="vla_replica_so101", is_simulated=True, control_hz=20,
            capabilities=frozenset({"seedable", "resettable", "privileged_success", "renderable"}),
            action_space=Box(shape=(6,), low=LOW, high=HIGH,
                semantics=ActionSemantics(control_mode="eef_abs_pose", rotation_repr="none",
                    gripper="continuous", frame="world", dim_labels=LABELS,
                    max_step=(.002, .002, .002, .03, .03, .05))),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(n, 256, 256) for n in ("top_camera", "wrist_camera")),
                state=StateSpec((StateField("eef_state", (6,)), StateField("joint_angles", (5,), "rad"),
                                 StateField("gripper_measured", (2,)),
                                 StateField("controller_feedback", (3,))))),
            docs=DOCS)

    def reset(self, scene: Scene, *, seed=None):
        self.close()
        path = (scene.metadata or {}).get("replica_scene_config", self.scene_config)
        if not path:
            raise ConfigError("Provide scene_config or use the vla_replica task")
        path = Path(path).resolve()
        self.config = read_scene(path)
        xml = path.with_suffix(".xml")
        if not xml.is_file():
            raise ConfigError(f"Compile the scene first: {path}")
        # Import simulator only after selecting the headless backend.
        os.environ.setdefault("MUJOCO_GL", "egl")
        from .controller import SO101Simulation
        from .scoring import ReplicaScorer
        self.sim = SO101Simulation(xml, width=256, height=256, max_steps=self.max_steps)
        self.scorer = ReplicaScorer(self.sim, self.config)
        self.instruction = self.config["instruction"]
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial_hash, self.trajectory, self.feedback = None, [], "ready"
        self.controller_feedback = np.array([1., 0., 0.])
        self.last = {"steps": 0, "success": False}
        if self.video_enabled:
            import imageio.v2 as imageio
            self.video = imageio.get_writer(str(self.output/"rollout.mp4"), fps=20,
                                            codec="libx264", macro_block_size=2)
        self.sim.on_step = self._after_tick
        self.snapshot = dict(scene_config=str(path), scene_json_sha256=sha256(path),
                             scene_xml_sha256=sha256(xml), scene=self.config,
                             asset_hashes={p.name: sha256(p) for p in sorted(
                                 Path(__file__).with_name("assets").joinpath("meshes").glob("*")
                                 ) if p.is_file()})
        xml_root = ET.parse(xml).getroot()
        compiler = xml_root.find("compiler")
        mesh_dir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
        if not mesh_dir.is_absolute():
            mesh_dir = xml.parent/mesh_dir
        self.snapshot["referenced_mesh_hashes"] = {}
        for mesh in xml_root.findall("./asset/mesh"):
            if mesh.get("file"):
                mesh_file = Path(mesh.get("file"))
                if not mesh_file.is_absolute():
                    mesh_file = mesh_dir/mesh_file
                self.snapshot["referenced_mesh_hashes"][str(mesh_file.resolve())] = sha256(mesh_file)
        shutil.copy2(xml, self.output/"scene.xml")
        shutil.copy2(path, self.output/"scene.json")
        (self.output/"scene_snapshot.json").write_text(json.dumps(self.snapshot, indent=2))
        observation = self._observation()
        self._frame(observation.images)
        if self.viewer_enabled:
            from .viewer import ViewerProcess
            self.viewer = ViewerProcess(self.sim)
        return observation

    def _after_tick(self):
        self.sim.success = self.scorer.update()
        self.last = dict(steps=self.sim.steps, success=self.sim.success,
                         scorer=self.scorer.diagnostics())
        state = self.sim.state()
        self.trajectory.append(dict(step=self.sim.steps, eef=state["eef_position_m"] +
                                    [state["pitch_rad"], state["roll_rad"], state["gripper_command"]]))
        if self.video is not None:
            self._frame(self.sim.render())
        if self.viewer is not None:
            self.viewer.sync()
            # Pace motion so live viewers can follow each 20 Hz control tick.
            delay = self.sim.control_dt-(time.monotonic()-self._tick_started)
            if delay > 0:
                time.sleep(delay)

    def _frame(self, images):
        if self.video is not None:
            self.video.append_data(np.concatenate(list(images.values()), axis=1))

    def _observation(self):
        state, images = self.sim.state(), self.sim.render()
        eef = np.array(state["eef_position_m"] + [state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
        if self.initial_hash is None:
            digest = hashlib.sha256()
            for name in sorted(images):
                digest.update(images[name].tobytes())
            digest.update(eef.tobytes())
            self.initial_hash = digest.hexdigest()
            from PIL import Image
            for name, pixels in images.items():
                Image.fromarray(pixels).save(self.output/f"initial_{name}.png")
        return Observation(images=images, state={"eef_state": eef,
            "joint_angles": np.array(list(state["joint_angles_rad"].values())),
            "gripper_measured": np.array([state["gripper_angle_rad"], state["fingertip_pad_center_distance_m"]]),
            "controller_feedback": self.controller_feedback.copy()},
            instruction=self.instruction, state_time=self.sim.steps/20,
            extra={"controller_feedback": self.feedback})

    def step(self, action):
        if self.viewer is not None and not self.viewer.is_running():
            return StepResult(self._observation(), terminated=True, termination_reason="viewer_closed",
                              info={"success": bool(self.last["success"])})
        self._tick_started = time.monotonic()
        if self.sim.steps >= self.max_steps:
            return StepResult(self._observation(), truncated=True, info={"success": bool(self.last["success"])})
        target = np.asarray(action.data, dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all() or np.any(target < LOW) or np.any(target > HIGH):
            self.feedback = "rejected: expected six finite in-bounds dimensions"
            self.controller_feedback = np.array([0., 0., 0.])
        else:
            seed = self.sim.data.qpos[self.sim.qadr].copy()
            current = self.sim.state()
            pose_change = (np.linalg.norm(target[:3]-current["eef_position_m"]) > 1e-6
                           or np.max(np.abs(target[3:5]-self.sim.orientation())) > 1e-6)
            if pose_change:
                q, p_error, r_error = self.sim.solve_ik(target[:3], target[3:5], seed)
            else:
                q, p_error, r_error = self.sim.data.ctrl[self.sim.act_ids].copy(), 0., 0.
            if p_error > self.sim.position_tolerance or r_error > self.sim.orientation_tolerance:
                self.feedback = f"IK rejected waypoint: position error {p_error:.4f} m, angle error {r_error:.3f} rad"
                self.controller_feedback = np.array([0., p_error, r_error])
            else:
                previous = self.sim.data.ctrl[self.sim.act_ids]
                limit = self.sim.joint_speed*self.sim.control_dt
                self.sim.data.ctrl[self.sim.act_ids] = np.clip(q, previous-limit, previous+limit)
                angle = self.sim.open_angle + target[5]*(self.sim.closed_angle-self.sim.open_angle)
                previous_grip = self.sim.data.ctrl[self.sim.gripper_act]
                self.sim.data.ctrl[self.sim.gripper_act] = np.clip(angle, previous_grip-limit, previous_grip+limit)
                self.sim.gripper_command = float(target[5])
                self.feedback = "waypoint applied; inspect measured pose for tracking error"
                self.controller_feedback = np.array([1., p_error, r_error])
        # A rejected command still advances time and the shared horizon.
        self.sim.step()
        success = self.scorer.success
        return StepResult(self._observation(), reward=float(success), terminated=success,
                          truncated=self.sim.steps >= self.max_steps and not success,
                          termination_reason="success" if success else None, info={"success": success})

    def close(self):
        try:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
        finally:
            try:
                if self.video is not None:
                    self.video.close()
                    self.video = None
            finally:
                if self.sim is not None:
                    try:
                        if self.scorer is not None:
                            (self.output/"score.json").write_text(json.dumps(self.scorer.diagnostics(), indent=2))
                    finally:
                        self.sim.close()
                        self.sim = None


def replica_embodiment(**kwargs):
    return ReplicaEmbodiment(**kwargs)

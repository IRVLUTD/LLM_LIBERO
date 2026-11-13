"""SO101 IK/rendering adapted from kochIRVL/sim/controller.py.

Modified for task-independent scoring and one 20 Hz tick per framework action.
The source repository is never imported or written at runtime.
"""

from pathlib import Path
import math

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares


SCENE = None
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
CAMERAS = ("top_camera", "wrist_camera")
BOUNDS = {"x": (0.04, 0.42), "y": (-0.24, 0.24), "z": (0.008, 0.40),
          "pitch": (-math.pi, math.pi), "roll": (-math.pi, math.pi), "gripper": (0.0, 1.0)}


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class SimulationStopped(RuntimeError):
    pass


class SO101Simulation:
    """Interpolate Cartesian waypoints, solve IK, then drive position actuators.

    pitch is the angle of the finger approach below the horizontal radial direction;
    pi/2 points down. roll rotates the jaws about that approach, with zero matching
    the home keyframe. Arm yaw follows the shoulder pan, rather than being a sixth
    independently commanded Cartesian dimension.
    """

    control_dt = 0.05
    position_tolerance = 0.004
    orientation_tolerance = 0.10
    linear_speed = 0.06
    angular_speed = 0.7
    joint_speed = 1.5
    open_angle = 1.0
    closed_angle = -0.174

    def __init__(self, scene=SCENE, *, width=640, height=480, max_steps=1800):
        self.scene = Path(scene).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.joint_ids = np.array([self.model.joint(n).id for n in ARM_JOINTS])
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.act_ids = np.array([self.model.actuator("act_" + n).id for n in ARM_JOINTS])
        self.gripper_qadr = self.model.joint("gripper").qposadr[0]
        self.gripper_act = self.model.actuator("act_gripper").id
        self.ee_id = self.model.site("ee_site").id
        self.joint_limits = self.model.jnt_range[self.joint_ids]
        self.width, self.height = width, height
        self.max_steps = max_steps
        self.steps = 0
        self.renderer = None
        self.on_step = None
        self.success = False
        self.success_dwell = 0.0
        self.gripper_command = 0.0
        base_id = self.model.body("Base").id
        self.arm_bodies = set()
        for body in range(1, self.model.nbody):
            parent = body
            while parent and parent != base_id:
                parent = self.model.body_parentid[parent]
            if parent == base_id:
                self.arm_bodies.add(body)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        # Settle physics with the existing home controls, without spending the episode budget.
        for _ in range(round(0.5 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        home_gripper = self.data.ctrl[self.gripper_act]
        self.gripper_command = float(np.clip(
            (self.open_angle - home_gripper) / (self.open_angle - self.closed_angle), 0, 1))

    def orientation(self, data=None):
        data = self.data if data is None else data
        rotation = data.site_xmat[self.ee_id].reshape(3, 3)
        origin = data.xanchor[self.joint_ids[0]]
        delta = data.site_xpos[self.ee_id] - origin
        yaw = math.atan2(delta[1], delta[0])
        radial = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        tangent = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        approach = -rotation[:, 1]
        pitch = math.atan2(-approach[2], np.dot(approach, radial))
        reference_z = np.cross(tangent, -approach)
        reference_z /= max(np.linalg.norm(reference_z), 1e-12)
        roll = math.atan2(np.dot(rotation[:, 0], reference_z), np.dot(rotation[:, 0], tangent))
        return np.array([pitch, roll])

    def state(self):
        pitch, roll = self.orientation()
        fixed = self.data.geom("fixed_jaw_pad_1").xpos
        moving = self.data.geom("moving_jaw_pad_1").xpos
        return {
            "simulation_time_s": round(float(self.data.time), 4),
            "control_steps": self.steps,
            "eef_position_m": self.data.site_xpos[self.ee_id].round(6).tolist(),
            "eef_rotation_matrix": self.data.site_xmat[self.ee_id].reshape(3, 3).round(6).tolist(),
            "reference_frames": self.reference_frames(),
            "pitch_rad": round(float(pitch), 6), "roll_rad": round(float(roll), 6),
            "joint_angles_rad": dict(zip(ARM_JOINTS, self.data.qpos[self.qadr].round(6).tolist())),
            "gripper_command": round(self.gripper_command, 4),
            "gripper_angle_rad": round(float(self.data.qpos[self.gripper_qadr]), 6),
            "fingertip_pad_center_distance_m": round(float(np.linalg.norm(fixed - moving)), 6),
        }

    def reference_frames(self):
        """Robot/camera calibration only; never include task-object state."""
        tool_rotation = self.data.site_xmat[self.ee_id].reshape(3, 3)
        base = self.data.body("Base")
        cameras = {}
        for name in CAMERAS:
            camera = self.model.camera(name)
            rotation = self.data.cam_xmat[camera.id].reshape(3, 3)
            focal = self.height / (2 * math.tan(math.radians(float(camera.fovy[0])) / 2))
            cameras[name] = {
                "position_world_m": self.data.cam_xpos[camera.id].round(6).tolist(),
                "R_world_from_camera": rotation.round(6).tolist(),
                "right_world": rotation[:, 0].round(6).tolist(),
                "up_world": rotation[:, 1].round(6).tolist(),
                "forward_world": (-rotation[:, 2]).round(6).tolist(),
                "intrinsics_px": {"fx": focal, "fy": focal,
                                  "cx": self.width / 2, "cy": self.height / 2},
                "image_size_px": [self.width, self.height],
            }
        return {
            "action_frame": "world", "position_units": "meters", "angle_units": "radians",
            "world": {"origin": "Base body origin on the table, fixed at [0,0,0]",
                      "axes": {"+x": "from base into workspace", "+y": "right-handed cross product: +z cross +x",
                               "+z": "up from the table"}, "table_z_m": 0.0},
            "base": {"position_world_m": base.xpos.round(6).tolist(),
                     "R_world_from_base": base.xmat.reshape(3, 3).round(6).tolist()},
            "tool": {"origin": "ee_site on fixed jaw near tips",
                     "R_world_from_tool": tool_rotation.round(6).tolist(),
                     "approach_direction_world": (-tool_rotation[:, 1]).round(6).tolist(),
                     "transform": "p_world = eef_position_m + R_world_from_tool @ p_tool"},
            "camera_convention": "local +X right, +Y up, -Z forward; image u right, v down; pixel centers at i+0.5,j+0.5",
            "projection": "p_camera = R_world_from_camera.T @ (p_world - camera_position); depth=-p_camera.z; u=fx*x/depth+cx; v=cy-fy*y/depth",
            "cameras": cameras,
        }

    def annotate_camera(self, pixels, name):
        """Compact world-axis compass; reserve little of the 256px observation."""
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        rotation = self.data.cam_xmat[self.model.camera(name).id].reshape(3, 3)
        h = pixels.shape[0]
        draw.rectangle((3, h-68, 92, h-3), fill=(20, 20, 20))
        draw.text((7, h-65), "WORLD", fill="white")
        origin = np.array([46., h-29.])
        for index, (label, color) in enumerate(zip(("X", "Y", "Z"),
                ((255, 80, 80), (80, 230, 100), (90, 150, 255)))):
            direction = np.array([rotation[index, 0], -rotation[index, 1]])
            end = origin + 18*direction
            draw.line([tuple(origin), tuple(end)], fill=color, width=2)
            norm = np.linalg.norm(direction)
            if norm > .15:
                unit = direction/norm
                side = np.array([-unit[1], unit[0]])
                draw.polygon([tuple(end), tuple(end-5*unit+2*side),
                              tuple(end-5*unit-2*side)], fill=color)
                draw.text(tuple(end+3*unit), label, fill=color)
        draw.rectangle((3, 3, 9+len(name)*7, 20), fill=(20, 20, 20))
        draw.text((7, 6), name, fill="white")
        return np.asarray(image).copy()

    def render(self, cameras=CAMERAS):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        images = {}
        option = mujoco.MjvOption()
        option.sitegroup[:] = 0
        for camera in cameras:
            # Sites are debugging markers, not part of the camera observation.
            self.renderer.update_scene(self.data, camera=camera,
                                       scene_option=option)
            images[camera] = self.annotate_camera(self.renderer.render(), camera)
        return images

    def solve_ik(self, position, orientation, seed, orientation_required=(True, True)):
        self.ik_data.qpos[:] = self.data.qpos
        seed = np.clip(seed, self.joint_limits[:, 0] + 1e-7, self.joint_limits[:, 1] - 1e-7)

        def residual(q):
            self.ik_data.qpos[self.qadr] = q
            mujoco.mj_fwdPosition(self.model, self.ik_data)
            return np.concatenate((self.ik_data.site_xpos[self.ee_id] - position,
                                   np.where(orientation_required, 0.04, 0.002)
                                   * wrap(self.orientation(self.ik_data) - orientation),
                                   1e-5 * (q - seed)))

        solution = least_squares(residual, seed,
                                 bounds=(self.joint_limits[:, 0] + 1e-7,
                                         self.joint_limits[:, 1] - 1e-7),
                                 max_nfev=80, ftol=1e-7, xtol=1e-7, gtol=1e-8)
        residual(solution.x)
        position_error = float(np.linalg.norm(self.ik_data.site_xpos[self.ee_id] - position))
        orientation_error = float(np.max(np.abs(wrap(self.orientation(self.ik_data) - orientation))
                                                * orientation_required))
        return solution.x, position_error, orientation_error

    def step(self):
        if self.steps >= self.max_steps:
            raise SimulationStopped("control-step budget exhausted")
        for _ in range(round(self.control_dt / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or any(w.number for w in self.data.warning):
            raise SimulationStopped("MuJoCo reported unstable physics")
        if self.on_step is not None:
            self.on_step()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

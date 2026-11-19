"""Adapted from kochIRVL/real_robot/kinematics.py; source repository stays read-only."""
"""URDF FK/IK and explicit LeRobot-degree <-> model-radian conversion.

Only the kinematic tree is read; no meshes, physics, or hardware are loaded.
"""

from pathlib import Path
from xml.etree import ElementTree as ET
import json
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
MOTORS = tuple(n + ".pos" for n in JOINTS) + ("gripper.pos",)
DEFAULT_PROFILE = Path(__file__).with_name("assets") / "profile.example.json"


def wrap(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def load_profile(path=DEFAULT_PROFILE):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    cfg["urdf"] = str((path.parent / cfg["urdf"]).resolve())
    for key in ("joint_signs", "joint_offsets_deg"):
        values = np.asarray(cfg[key], dtype=float)
        if values.shape != (5,) or not np.isfinite(values).all():
            raise ValueError(f"{key} requires five finite numbers")
    if not all(s in (-1, 1) for s in cfg["joint_signs"]):
        raise ValueError("Joint signs must be +1 or -1")
    for name in ("world_from_base_xyz_m", "world_from_base_rpy_rad", "tip_from_tool_xyz_m", "tip_from_tool_rpy_rad"):
        values = np.asarray(cfg[name], dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"{name} requires three finite numbers")
    for name in ("linear_speed_m_s", "angular_speed_rad_s", "joint_speed_deg_s", "control_hz",
                 "gripper_speed_units_s", "max_translation_m", "max_tracking_error_deg",
                 "position_tolerance_m", "observation_max_age_s"):
        if not isinstance(cfg[name], (float, int)) or not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("x", "y", "z"):
        bounds = np.asarray(cfg["workspace_m"][name], dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid workspace bounds for {name}")
    for name in ("gripper_open_motor", "gripper_closed_motor"):
        if not 0 <= cfg[name] <= 100:
            raise ValueError(f"{name} must be in [0,100]")
    if cfg["gripper_open_motor"] == cfg["gripper_closed_motor"]:
        raise ValueError("Open and closed gripper endpoints must differ")
    return cfg


class Kinematics:
    def __init__(self, config):
        self.config = config
        self.signs = np.asarray(config["joint_signs"], dtype=float)
        self.offsets = np.asarray(config["joint_offsets_deg"], dtype=float)
        self.world_from_base = transform(config["world_from_base_xyz_m"], config["world_from_base_rpy_rad"])
        self.tip_from_tool = transform(config["tip_from_tool_xyz_m"], config["tip_from_tool_rpy_rad"])
        root = ET.parse(config["urdf"]).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], config["tip_link"]
        visited = set()
        while link != config["base_link"]:
            if link in visited or link not in by_child:
                raise ValueError("Tip does not have a valid chain to the configured base link")
            visited.add(link)
            joint = by_child[link]
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            axis = joint.find("axis")
            direction = np.fromstring(axis.get("xyz", "1 0 0"), sep=" ") if axis is not None else np.array([1., 0, 0])
            chain.append((joint, transform(xyz, rpy), direction))
            link = joint.find("parent").get("link")
        self.chain = list(reversed(chain))
        active = [j.get("name") for j, _, _ in self.chain if j.get("type") != "fixed"]
        if active != list(JOINTS):
            raise ValueError(f"Expected SO101 arm joints {JOINTS}; found {active}")
        limits = []
        for joint, _, _ in self.chain:
            if joint.get("type") == "fixed":
                continue
            if joint.get("type") != "revolute":
                raise ValueError("This controller supports bounded revolute arm joints")
            limit = joint.find("limit")
            limits.append([float(limit.get("lower")), float(limit.get("upper"))])
        self.limits = np.asarray(limits)
        for name, bounds in config.get("joint_limit_overrides_deg", {}).items():
            values = np.asarray(bounds, dtype=float)
            if name not in JOINTS or values.shape != (2,) or not np.isfinite(values).all() or values[0] >= values[1]:
                raise ValueError(f"Invalid verified model-joint limit override for {name}")
            self.limits[JOINTS.index(name)] = np.deg2rad(values)

    def motor_to_model(self, motors):
        values = np.asarray([motors[n + ".pos"] for n in JOINTS], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite measured motor positions")
        return np.deg2rad(self.signs * values + self.offsets)

    def model_to_motor(self, q, gripper_motor):
        values = (np.rad2deg(q) - self.offsets) / self.signs
        return {**dict(zip((n + ".pos" for n in JOINTS), values.tolist())), "gripper.pos": float(gripper_motor)}

    def forward(self, q):
        matrix = self.world_from_base.copy()
        index = 0
        for joint, origin, axis in self.chain:
            matrix = matrix @ origin
            if joint.get("type") != "fixed":
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * q[index]).as_matrix()
                matrix = matrix @ rotation
                index += 1
        return matrix @ self.tip_from_tool

    def orientation(self, pose):
        # Interpret pitch/roll in the configured world frame, matching the sim
        # convention after aligning the URDF tip frame to the tool frame.
        delta = pose[:3, 3] - self.world_from_base[:3, 3]
        yaw = math.atan2(delta[1], delta[0])
        radial = np.array([math.cos(yaw), math.sin(yaw), 0])
        tangent = np.array([-math.sin(yaw), math.cos(yaw), 0])
        approach = -pose[:3, 1]
        pitch = math.atan2(-approach[2], np.dot(approach, radial))
        reference_z = np.cross(tangent, -approach)
        reference_z /= max(np.linalg.norm(reference_z), 1e-12)
        roll = math.atan2(np.dot(pose[:3, 0], reference_z), np.dot(pose[:3, 0], tangent))
        return np.array([pitch, roll])

    def solve(self, position, orientation, seed, required, motor_limits):
        # Intersect URDF limits with the actual calibrated motor travel.
        normalized_limits = np.deg2rad(self.signs[:, None] * motor_limits[:5] + self.offsets[:, None])
        lower = np.maximum(self.limits[:, 0], normalized_limits.min(axis=1)) + 1e-7
        upper = np.minimum(self.limits[:, 1], normalized_limits.max(axis=1)) - 1e-7
        if np.any(lower >= upper):
            raise ValueError("Model limits and calibrated motor limits do not overlap")
        seed = np.clip(seed, lower, upper)

        def residual(q):
            pose = self.forward(q)
            return np.concatenate((pose[:3, 3] - position,
                np.where(required, 0.04, 0.002) * wrap(self.orientation(pose) - orientation),
                1e-5 * (q - seed)))

        result = least_squares(residual, seed, bounds=(lower, upper), max_nfev=100,
                               ftol=1e-8, xtol=1e-8, gtol=1e-9)
        pose = self.forward(result.x)
        p_error = float(np.linalg.norm(pose[:3, 3] - position))
        r_error = float(np.max(np.abs(wrap(self.orientation(pose) - orientation)) * required))
        return result.x, p_error, r_error


def calibrated_motor_limits(calibration):
    limits = []
    for name in JOINTS:
        entry = calibration[name]
        span = float(entry["range_max"] - entry["range_min"])
        if span <= 0:
            raise ValueError(f"Invalid calibration travel for {name}")
        half_degrees = span / 2 * 360 / 4095
        limits.append([-half_degrees, half_degrees])
    limits.append([0, 100])
    return np.asarray(limits)

"""Read-only hardware settings; importing this module never connects devices."""
import json
import math
from pathlib import Path
import sys

ASSETS = Path(__file__).with_name("assets")
DEFAULT_CONFIG = ASSETS/"mock_config.json"
MOTOR_KEYS = tuple(n+".pos" for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))


def load_settings(path=None):
    path = Path(path or DEFAULT_CONFIG).resolve()
    settings = json.loads(path.read_text())
    if settings.get("schema_version") != 1:
        raise ValueError("Hardware configuration requires schema_version=1")
    for field in ("profile_file", "calibration_file", "reset_action_file"):
        if settings.get(field):
            resource = Path(settings[field])
            settings[field] = str((resource if resource.is_absolute() else path.parent/resource).resolve())
            if not Path(settings[field]).is_file():
                raise ValueError(f"Missing {field}: {settings[field]}")
    if not settings.get("profile_file") or not settings.get("calibration_file"):
        raise ValueError("Specify profile_file and calibration_file")
    settings.setdefault("python", sys.executable)
    settings.setdefault("preview", True)
    settings.setdefault("video", True)
    settings.setdefault("width", 640)
    settings.setdefault("height", 480)
    settings.setdefault("fps", 30)
    settings.setdefault("zoom", 1.5)
    settings.setdefault("reset_mode", "none")
    settings.setdefault("reset_max_steps", 1800)
    if settings["reset_mode"] not in ("fixed", "none"):
        raise ValueError("Hardware reset_mode must be fixed or none")
    if settings["reset_mode"] == "fixed" and not settings.get("reset_action_file"):
        raise ValueError("Fixed reset requires reset_action_file")
    for name in ("width", "height", "fps", "reset_max_steps"):
        value = settings[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(settings["zoom"], (int, float)) or not math.isfinite(settings["zoom"]) or settings["zoom"] < 1:
        raise ValueError("zoom must be finite and >=1")
    for name in ("preview", "video"):
        if not isinstance(settings[name], bool):
            raise ValueError(f"{name} must be a boolean")
    if not settings.get("port") or not settings.get("follower_id"):
        raise ValueError("Specify port and follower_id")
    if settings.get("top_index") is None or settings.get("wrist_index") is None:
        raise ValueError("Specify top_index and wrist_index")
    if settings["top_index"] == settings["wrist_index"]:
        raise ValueError("Top and wrist cameras must identify different devices")
    profile_path = Path(settings["profile_file"])
    profile = json.loads(profile_path.read_text())
    urdf = Path(profile["urdf"])
    profile["urdf"] = str((urdf if urdf.is_absolute() else profile_path.parent/urdf).resolve())
    if not Path(profile["urdf"]).is_file():
        raise ValueError(f"Missing URDF: {profile['urdf']}")
    if settings["reset_mode"] == "fixed":
        reset = json.loads(Path(settings["reset_action_file"]).read_text())
        if set(reset) != set(MOTOR_KEYS) or any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v) for v in reset.values()):
            raise ValueError("Reset requires exactly six finite LeRobot .pos values")
    return settings, profile


def execution_gate(settings, profile, mode, assume_calibration=False, *, requires_cartesian=True):
    if mode not in ("mock", "inspect", "dry-run", "execute"):
        raise ValueError("Unknown hardware mode")
    if mode == "execute" and requires_cartesian and not assume_calibration and (
            profile.get("kinematics_verified") is not True or profile.get("world_frame_verified") is not True):
        raise ValueError("Execution requires verified kinematics/world frame or explicit --assume-calibration; no devices were connected")
    if mode != "mock" and Path(settings["calibration_file"]).stem != settings["follower_id"]:
        raise ValueError("calibration_file basename must match follower_id")

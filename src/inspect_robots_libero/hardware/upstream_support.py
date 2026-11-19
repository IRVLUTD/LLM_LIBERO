"""Pinned upstream adapter configuration and source provenance."""
import hashlib
from importlib import metadata
import json
from pathlib import Path

from .kinematics import calibrated_motor_limits


def upstream_provenance():
    """Require the actual, unmodified upstream source revision used by this benchmark."""
    try:
        import inspect_robots_so101
    except ImportError as exc:
        raise RuntimeError("Install requirements.so101-upstream.txt in the evaluation and worker environments") from exc
    pin = json.loads((Path(__file__).with_name("assets")/"upstream_so101.json").read_text())
    root = Path(inspect_robots_so101.__file__).parent
    actual = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in pin["source_sha256"]}
    if actual != pin["source_sha256"]:
        raise RuntimeError(f"inspect-robots-so101 source differs from pinned commit {pin['commit']}; install requirements.so101-upstream.txt")
    return dict(pin, installed_version=metadata.version("inspect-robots-so101"),
                embodiment_class="inspect_robots_so101.embodiment.SOArmEmbodiment", source_directory=str(root))


def upstream_config(settings, profile):
    """Use calibrated native joint units and the existing hardware image resolution."""
    from inspect_robots_so101.config import SOArmConfig
    calibration = json.loads(Path(settings["calibration_file"]).read_text())
    limits = calibrated_motor_limits(calibration)
    return SOArmConfig(port=settings["port"], robot_id=settings["follower_id"],
        calibration_dir=str(Path(settings["calibration_file"]).parent),
        cameras=("top", "wrist"), control_hz=profile["control_hz"],
        cam_width=settings["width"], cam_height=settings["height"],
        joint_low=tuple(limits[:, 0]), joint_high=tuple(limits[:, 1]),
        home_pose=None, use_degrees=True, joints_are_delta=False,
        disable_torque_on_disconnect=False,
        settle_tolerance=settings.get("upstream_settle_tolerance"),
        settle_timeout_s=settings.get("upstream_settle_timeout_s", 1.),
        settle_timeout_budget=settings.get("upstream_settle_timeout_budget", 20))

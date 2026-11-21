"""Explicit post-episode torque release; never sends position goals or enables torque."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .settings import load_settings, execution_gate, MOTOR_KEYS


def _make_robot(settings):
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    return SO101Follower(SO101FollowerConfig(port=settings["port"],id=settings["follower_id"],
        calibration_dir=Path(settings["calibration_file"]).parent,cameras={},use_degrees=True,
        max_relative_target=None,disable_torque_on_disconnect=False))


def release_torque(config_path, output):
    """Connect directly to the bus, verify selected calibration, disable and read back.

    SOFollower.connect/configure are deliberately bypassed: those may enable
    torque, move the arm, or overwrite servo tuning. The serial lock is shared
    with the evaluation worker. This helper must only be invoked after explicit
    operator confirmation, and after closing that worker.
    """
    report=dict(requested_utc=datetime.now(timezone.utc).isoformat(),status="failed",torque_disabled=None,
                motor_position_writes=0,torque_on_writes=0,torque_disable_requests=0)
    bus=None
    lock=None
    try:
        settings,profile=load_settings(config_path)
        execution_gate(settings,profile,"execute",requires_cartesian=False)
        report.update(port=settings["port"],follower_id=settings["follower_id"])
        port_key=hashlib.sha256(os.path.realpath(settings["port"]).encode()).hexdigest()[:24]
        lock=(Path(tempfile.gettempdir())/f"llm_libero_so101_{port_key}.lock").open("a")
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Serial port is still owned by another hardware worker") from None
        bus=_make_robot(settings).bus
        bus.connect()
        if set(bus.motors)!={name.removesuffix(".pos") for name in MOTOR_KEYS}:
            raise ValueError("Expected the six SO-101 motors; no torque write attempted")
        if not bus.is_calibrated:
            raise ValueError("Motor EEPROM calibration does not match the selected robot; no torque write attempted")
        report["torque_disable_requests"]=1
        bus.disable_torque()
        states={name:int(bus.read("Torque_Enable",name)) for name in bus.motors}
        report["torque_enable_readback"]=states
        if not states or any(value!=0 for value in states.values()):
            raise RuntimeError("Could not confirm torque disabled on every motor")
        report.update(status="disabled",torque_disabled=True)
    except Exception as exc:
        report.update(error_type=type(exc).__name__,detail=str(exc))
    finally:
        if bus is not None and bus.is_connected:
            try:
                bus.disconnect(disable_torque=False)
            except Exception as exc:
                report["disconnect_error"]=str(exc)
        if lock is not None:
            lock.close()
        path=Path(output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+"\n")
    return report

"""Offline Cartesian replay planning. Importing this module never opens devices."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .kinematics import Kinematics, MOTORS, calibrated_motor_limits, wrap

LABELS = ("x", "y", "z", "pitch", "roll", "gripper")


class ReplayRejected(ValueError):
    def __init__(self, reason, **details):
        super().__init__(reason)
        self.report = dict(accepted=False, reason=reason, **details)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recording(source, track="commands", limit=None, *, start_step=1):
    """Read either requested Cartesian action records or achieved Cartesian poses.

    This does not interpret simulator joint radians as hardware motor degrees.
    Commands can include targets the simulator's IK rejected. The observed track
    is a different experiment from replaying the commands.
    """
    source = Path(source).resolve()
    if (source/"evaluation.json").is_file():
        evaluation = json.loads((source/"evaluation.json").read_text())
        runs = [r for r in evaluation["runs"] if r.get("benchmark") == "replica"]
        if len(runs) != 1:
            raise ValueError("Select a single episode directory when the evaluation contains multiple simulation runs")
        row = runs[0]
        source = source/row["run_folder"]
        hz = evaluation["protocol"]["control_hz"]
    else:
        row, hz = {}, 20
        evaluation_path = source.parent/"evaluation.json"
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text())
            row = next((r for r in evaluation["runs"] if r["run_folder"] == source.name), {})
            hz = evaluation["protocol"]["control_hz"]
    scene_path = source/"sim/scene.json"
    scene = json.loads(scene_path.read_text())
    if track == "commands":
        files = list((source/"inspect/actions").rglob("*.jsonl"))
        if len(files) != 1:
            raise ValueError("Expected exactly one action stream in the source episode")
        path = files[0]
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not records or records[0].get("kind") != "header" or records[0].get("labels") != list(LABELS):
            raise ValueError("Recording must contain Cartesian x,y,z,pitch,roll,gripper actions")
        records = records[1:]
        if any(r.get("t") != i for i, r in enumerate(records)):
            raise ValueError("Action indices must be contiguous, starting at zero")
        targets = [r["action"] for r in records]
    elif track == "observed":
        path = source/"trajectory.json"
        records = json.loads(path.read_text())
        if any(r.get("step") != i+1 for i, r in enumerate(records)):
            raise ValueError("Observed trajectory steps must be contiguous, starting at one")
        targets = [r["eef"] for r in records]
    else:
        raise ValueError("track must be commands or observed")
    if not isinstance(start_step,int) or not 1 <= start_step <= len(targets):
        raise ValueError("source start step must select an existing one-based recording step")
    targets=targets[start_step-1:]
    if limit is not None:
        if limit < 1:
            raise ValueError("source step limit must be positive")
        targets = targets[:limit]
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 6 or not len(targets) or not np.isfinite(targets).all():
        raise ValueError("Recording must contain a nonempty finite N-by-6 Cartesian path")
    score_path = source/"sim/score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    return targets, dict(episode=str(source), scene_file=str(scene_path), track=track,
        recording_file=str(path), recording_sha256=file_hash(path), scene_sha256=file_hash(scene_path),
        source_hz=hz, source_steps=len(targets), source_start_step=start_step, source_end_step=start_step+len(targets)-1,
        source_success=score.get("success", row.get("environment_success")),
        source_fidelity=scene.get("fidelity"), scene_id=scene["id"], instruction=scene["instruction"],
        source_objects=scene.get("objects", []), reference_image=scene.get("provenance", {}).get("reference_image"))


def map_targets(targets, frame):
    """Explicit Z-up rigid mapping; yaw rotates the world, not tool roll/pitch."""
    if frame.get("schema_version") != 1 or not isinstance(frame.get("verified"), bool):
        raise ValueError("Frame map needs schema_version=1 and a boolean verified field")
    shift = np.asarray(frame.get("translation_m"), dtype=float)
    yaw = frame.get("yaw_deg")
    if shift.shape != (3,) or not np.isfinite(shift).all() or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
        raise ValueError("Frame map needs three finite translation_m values and finite yaw_deg")
    radians = math.radians(yaw)
    rotation = np.array([[math.cos(radians), -math.sin(radians), 0],
                         [math.sin(radians), math.cos(radians), 0], [0, 0, 1]])
    mapped = np.asarray(targets, dtype=float).copy()
    mapped[:, :3] = mapped[:, :3] @ rotation.T+shift
    return mapped


def identity_frame():
    return dict(schema_version=1, verified=False, translation_m=[0, 0, 0], yaw_deg=0,
                description="Unverified identity mapping from simulation table/world coordinates to hardware profile world coordinates; no automatic alignment")


def retarget_pitch_height(profile, calibration, targets, start_motors, *, source_start_step=1):
    """Explicitly adapt pitch and low heights; preserve X/Y, roll and gripper.

    This is an adjusted trajectory, not an exact recording replay. Leave a joint
    margin in the feasibility solve and a Cartesian margin above the height floor
    so the subsequent strict whole-path planner can validate the interpolations.
    """
    original=np.asarray(targets,dtype=float)
    if original.ndim!=2 or original.shape[1]!=6 or not len(original) or not np.isfinite(original).all():
        raise ReplayRejected("Retargeting needs a nonempty finite N-by-6 Cartesian path")
    kin=Kinematics(profile)
    limits=calibrated_motor_limits(calibration)
    kin.limits[:,0]+=math.radians(1)
    kin.limits[:,1]-=math.radians(1)
    seed=kin.motor_to_model(start_motors)
    adjusted=original.copy()
    minimum_z=profile["workspace_m"]["z"][0]+profile["position_tolerance_m"]+.001
    adjusted[:,2]=np.maximum(adjusted[:,2],minimum_z)
    changes=[]
    for i,target in enumerate(adjusted):
        seed,pe,_=kin.solve(target[:3],target[3:5],seed,np.array([False,True]),limits)
        pose=kin.forward(seed)
        pitch,roll=kin.orientation(pose)
        if pe>profile["position_tolerance_m"] or abs(wrap(roll-target[4]))>.08:
            raise ReplayRejected("Cannot retarget this position while retaining tool roll",source_step=source_start_step+i,
                                 position_error_m=float(pe),roll_error_rad=float(wrap(roll-target[4])))
        target[3]=pitch
        changes.append(dict(source_step=source_start_step+i,original_eef=original[i].tolist(),
                            adjusted_eef=target.tolist(),height_change_m=float(target[2]-original[i,2]),
                            pitch_change_deg=float(math.degrees(wrap(target[3]-original[i,3])))))
    heights=adjusted[:,2]-original[:,2]
    pitch_deltas=np.rad2deg(wrap(adjusted[:,3]-original[:,3]))
    summary=dict(mode="pitch-height",height_adjusted_samples=int(np.sum(heights>1e-9)),
                 max_height_increase_m=float(np.max(heights)),minimum_target_height_m=float(np.min(adjusted[:,2])),
                 pitch_adjusted_samples=int(np.sum(np.abs(pitch_deltas)>1e-4)),
                 max_pitch_change_deg=float(np.max(np.abs(pitch_deltas))),
                 preserved_dimensions=["x","y","roll","gripper"],joint_margin_deg=1,
                 note="Adjusted path; strict whole-path hardware preflight is still required")
    return adjusted,dict(summary=summary,changes=changes)


def plan_replay(profile, calibration, targets, start_motors, *, source_hz=20,
                time_scale=2., max_steps=12000, source_start_step=1, check_stop=lambda: None):
    """Validate the WHOLE replay before movement, retiming without clipping targets.

    Every Cartesian sample is solved with hardware IK; native motor subdivisions
    enforce physical travel and speed limits. A bounded approach from the supplied
    start pose is part of the plan. Long replay travel is intentional, outside the
    LLM's per-call translation limit; all workspace/tracking/stop guards remain.
    """
    if not math.isfinite(source_hz) or source_hz <= 0 or not math.isfinite(time_scale) or time_scale < 1:
        raise ValueError("source_hz must be positive and time_scale must be finite and >=1")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(source_start_step,int) or source_start_step < 1:
        raise ValueError("source_start_step must be a positive integer")
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 6 or not len(targets) or not np.isfinite(targets).all():
        raise ReplayRejected("Expected a nonempty finite N-by-6 Cartesian path")
    low = np.array([profile["workspace_m"][a][0] for a in ("x", "y", "z")]+[-math.pi, -math.pi, 0])
    high = np.array([profile["workspace_m"][a][1] for a in ("x", "y", "z")]+[math.pi, math.pi, 1])
    bad = np.argwhere((targets < low) | (targets > high))
    if len(bad):
        issues = [dict(source_step=int(i)+source_start_step, dimension=LABELS[j], value=float(targets[i,j]),
                       allowed=[float(low[j]), float(high[j])]) for i,j in bad[:20]]
        raise ReplayRejected("Recorded targets exceed the hardware workspace; no targets were clipped or shifted",
                             violation_count=len(bad), issues=issues)
    kin = Kinematics(profile)
    motor_limits = calibrated_motor_limits(calibration)
    previous = np.array([start_motors[n] for n in MOTORS], dtype=float)

    def validate_motor(vector):
        if not np.isfinite(vector).all() or np.any(vector < motor_limits[:,0]-.09) or np.any(vector > motor_limits[:,1]+.09):
            raise ReplayRejected("Replay exceeds calibrated motor travel")

    validate_motor(previous)
    seed = kin.motor_to_model(start_motors)
    if np.any(seed < kin.limits[:,0]) or np.any(seed > kin.limits[:,1]):
        raise ReplayRejected("Start pose exceeds hardware model joint limits")
    pose = kin.forward(seed)
    opened, closed = profile["gripper_open_motor"], profile["gripper_closed_motor"]
    start = np.r_[pose[:3,3], kin.orientation(pose), np.clip((previous[5]-opened)/(closed-opened),0,1)]
    if np.any(start < low) or np.any(start > high):
        raise ReplayRejected("Start pose exceeds hardware workspace", start_eef=start.tolist())
    dt = 1/profile["control_hz"]
    speed = np.array([profile["joint_speed_deg_s"]]*5+[profile["gripper_speed_units_s"]])
    commands, indices, approach_steps = [], [], 0
    for index, target in enumerate(targets):
        check_stop()
        delta = target-start
        delta[3:5] = wrap(delta[3:5])
        duration = max(np.linalg.norm(delta[:3])/profile["linear_speed_m_s"],
                       np.max(np.abs(delta[3:5]))/profile["angular_speed_rad_s"],
                       abs(delta[5]*(closed-opened))/profile["gripper_speed_units_s"],
                       dt if index == 0 else time_scale/source_hz)
        ticks = max(1, math.ceil(duration/dt-1e-9))
        if len(commands)+ticks > max_steps:
            raise ReplayRejected("Replay exceeds the motor-step budget", source_step=index+source_start_step)
        for fraction in np.linspace(1/ticks,1,ticks):
            check_stop()
            point = start+fraction*delta
            point[3:5] = wrap(point[3:5])
            seed, pe, re = kin.solve(point[:3],point[3:5],seed,np.array([True,True]),motor_limits)
            if pe > profile["position_tolerance_m"] or re > .1:
                raise ReplayRejected("Hardware IK cannot reproduce the recorded path", source_step=index+source_start_step,
                    position_error_m=float(pe), angle_error_rad=float(re), rejected_eef=point.tolist())
            vector = np.array(list(kin.model_to_motor(seed,opened+point[5]*(closed-opened)).values()))
            validate_motor(vector)
            count = max(1, math.ceil(float(np.max(np.abs(vector-previous)/(speed*dt)))-1e-9))
            if len(commands)+count > max_steps:
                raise ReplayRejected("Retimed replay exceeds the motor-step budget", source_step=index+source_start_step)
            for alpha in np.linspace(1/count,1,count):
                motor = previous+alpha*(vector-previous)
                measured = kin.forward(kin.motor_to_model(dict(zip(MOTORS,motor))))[:3,3]
                if np.any(measured < low[:3]) or np.any(measured > high[:3]):
                    raise ReplayRejected("Interpolated hardware joint path leaves the Cartesian workspace",source_step=index+source_start_step)
                commands.append(motor.tolist())
                indices.append(index+source_start_step)
                if index == 0:
                    approach_steps += 1
            previous = vector
        start = target.copy()
    return dict(schema_version=1, accepted=True, motor_targets=commands, source_steps=indices,
        start_motors=start_motors, summary=dict(source_samples=len(targets), motor_steps=len(commands),
            source_start_step=source_start_step,source_end_step=source_start_step+len(targets)-1,
            approach_motor_steps=approach_steps, control_hz=profile["control_hz"],
            planned_motion_seconds=len(commands)*dt, source_motion_seconds=len(targets)/source_hz,
            requested_time_scale=time_scale, timing="Retimed to hardware speeds; inference pauses omitted",
            initial_pose_transition="Bounded Cartesian approach from current pose; no fixed startup reset",
            real_api_calls=0, agent_cost_usd=0))

"""Preflight/replay a recorded Replica Cartesian path without calling an LLM.

Default mode prepares files offline. Execution requires --mode execute, a full
hardware preflight and an interactive operator start. Source targets are never
silently clipped, shifted or interpreted as hardware motor angles.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

from inspect_robots_libero.hardware.kinematics import load_profile
from inspect_robots_libero.hardware.settings import load_settings
from inspect_robots_libero.hardware.replay import (
    ReplayRejected, file_hash, identity_frame, load_recording, map_targets, plan_replay, retarget_pitch_height,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True,help="Evaluation root with one simulation run, or exact episode directory")
    parser.add_argument("--hardware-config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",choices=("prepare","mock","dry-run","execute"),default="prepare")
    parser.add_argument("--track",choices=("commands","observed"),default="commands")
    parser.add_argument("--frame-map",type=Path,help="Explicit yaw/translation mapping JSON; default identity is unverified")
    parser.add_argument("--time-scale",type=float,default=2,help="Minimum time per source tick relative to simulation; >=1, default 2")
    parser.add_argument("--retarget",choices=("none","pitch-height"),default="none",help="Explicitly adjust pitch and low heights to hardware limits")
    parser.add_argument("--start-source-step",type=int,default=1,help="One-based source step to resume at; default 1")
    parser.add_argument("--max-source-steps",type=int,help="Maximum source samples starting at start-source-step; never skip individual invalid waypoints")
    parser.add_argument("--max-steps",type=int,default=12000,help="Total motor commands, including initial approach")
    parser.add_argument("--policy-seconds",type=float,default=900)
    parser.add_argument("--start-motors",type=Path,help="Offline preflight start pose; defaults to config's reset_action_file")
    parser.add_argument("--assume-calibration",action="store_true")
    parser.add_argument("--no-video",action="store_true")
    args=parser.parse_args(argv)
    settings,_=load_settings(args.hardware_config)
    profile=load_profile(settings["profile_file"])
    calibration=json.loads(Path(settings["calibration_file"]).read_text())
    targets,source=load_recording(args.source,args.track,args.max_source_steps,start_step=args.start_source_step)
    frame=json.loads(args.frame_map.read_text()) if args.frame_map else identity_frame()
    targets=map_targets(targets,frame)
    if args.mode=="execute" and not frame["verified"] and not args.assume_calibration:
        parser.error("Physical replay needs a verified frame map or explicit --assume-calibration")
    if args.mode=="execute" and not sys.stdin.isatty():
        parser.error("Physical replay needs an interactive terminal for operator start")
    start_file=args.start_motors or settings.get("reset_action_file")
    if not start_file:
        parser.error("Provide --start-motors for offline preflight when configuration has no reset_action_file")
    start_motors=json.loads(Path(start_file).read_text())
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/"frame_map.json").write_text(json.dumps(frame,indent=2))
    (output/"mapped_targets.json").write_text(json.dumps(targets.tolist()))
    with (output/"source_targets.csv").open("w",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(["source_step","x_m","y_m","z_m","pitch_rad","roll_rad","gripper_closure"])
        writer.writerows([i+args.start_source_step,*point] for i,point in enumerate(targets.tolist()))
    retargeting=None
    if args.retarget=="pitch-height":
        try:
            targets,retarget_report=retarget_pitch_height(profile,calibration,targets,start_motors,
                                                        source_start_step=args.start_source_step)
        except ReplayRejected as exc:
            (output/"retarget_report.json").write_text(json.dumps(exc.report,indent=2))
            print(f"Retargeting blocked before connecting hardware: {exc}")
            return 2
        retargeting=retarget_report["summary"]
        (output/"retarget_report.json").write_text(json.dumps(retarget_report,indent=2))
        (output/"retargeted_targets.json").write_text(json.dumps(targets.tolist()))
        print("Adjusted trajectory:",json.dumps(retargeting))
    mapped_objects=[]
    for obj in source["source_objects"]:
        point=map_targets([obj["pos"]+[0,0,0]],frame)[0,:3].tolist()
        mapped_objects.append(dict(name=obj["name"],kind=obj["kind"],source_origin_m=obj["pos"],
                                   hardware_world_origin_m=point,size=obj.get("size")))
    manifest=dict(source=source,frame_map=frame,source_track=args.track,
        retargeting=retargeting,
        mapped_object_origins=mapped_objects,
        hardware_config=str(args.hardware_config.resolve()),hardware_config_sha256=file_hash(args.hardware_config),
        profile_file=settings["profile_file"],profile_sha256=file_hash(settings["profile_file"]),
        calibration_sha256=file_hash(settings["calibration_file"]),
        urdf_sha256=file_hash(profile["urdf"]),start_motors_file=str(Path(start_file).resolve()),
        offline_start_sha256=file_hash(start_file),time_scale=args.time_scale,
        replay_script_sha256=file_hash(__file__),replay_planner_sha256=file_hash(plan_replay.__code__.co_filename),
        mode=args.mode,real_api_calls=0,agent_cost_usd=0,
        notes=["Recorded Cartesian targets are re-solved with hardware IK, not copied simulator joint values.",
               "Speed limits retime motion; original inference pauses are omitted.",
               "There is no automatic scene alignment, collision avoidance, or success oracle.",
               "The initial approach is planned from the current arm pose; fixed startup reset is disabled."])
    (output/"replay_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"Source: {source['scene_id']}; {len(targets)} {args.track} samples at {source['source_hz']} Hz")
    print(f"Recorded simulation success: {source['source_success']}; API calls/cost for replay: 0 / $0")
    try:
        plan=plan_replay(profile,calibration,targets,start_motors,source_hz=source["source_hz"],
                         time_scale=args.time_scale,max_steps=args.max_steps,source_start_step=args.start_source_step)
        report=dict(accepted=True,**plan["summary"],start_source="offline supplied motor pose; physical mode replans from measured state")
        (output/"offline_plan.json").write_text(json.dumps(plan))
        with (output/"motor_targets.csv").open("w",newline="") as file:
            from inspect_robots_libero.hardware.kinematics import MOTORS
            writer=csv.writer(file)
            writer.writerow(["motor_step","source_step",*MOTORS])
            writer.writerows([i+1,plan["source_steps"][i],*point] for i,point in enumerate(plan["motor_targets"]))
    except ReplayRejected as exc:
        report=exc.report
    (output/"offline_preflight.json").write_text(json.dumps(report,indent=2))
    if not report["accepted"]:
        print(f"BLOCKED before connecting hardware: {report['reason']}")
        print(f"Details: {output/'offline_preflight.json'}")
        return 2
    print(f"Offline preflight: {plan['summary']['motor_steps']} motor steps, including approach; at least {plan['summary']['planned_motion_seconds']:.1f} s")
    if args.mode=="prepare":
        print(f"Prepared offline only: {output}")
        return 0
    from inspect_robots.scene import Scene
    from inspect_robots_libero.hardware.embodiment import SO101HardwareEmbodiment
    # A local derived config prevents any fixed reset before replay preflight.
    replay_config=output/"hardware_config.json"
    settings.update(reset_mode="none",video=not args.no_video and settings["video"])
    replay_config.write_text(json.dumps(settings,indent=2))
    env=SO101HardwareEmbodiment(config_path=replay_config,mode=args.mode,output_dir=output/"hardware",
        max_steps=args.max_steps,policy_seconds=args.policy_seconds,auto_start=True,
        assume_calibration=args.assume_calibration,video=not args.no_video)
    try:
        env.reset(Scene(id=source["scene_id"],instruction=source["instruction"],
                        metadata={"replica_scene_config":source["scene_file"]}))
        prepared=env._rpc(dict(op="replay_plan",targets=targets.tolist(),source_hz=source["source_hz"],time_scale=args.time_scale,
                               source_start_step=args.start_source_step))
        if not prepared["accepted"]:
            print(f"Live-state preflight rejected; no trajectory sent: {prepared['reason']}")
            return 2
        print(f"Live-state preflight: {prepared['motor_steps']} motor steps; {prepared['planned_motion_seconds']:.1f} s minimum")
        if args.mode=="dry-run":
            print("Read-only dry-run complete; no motor targets or torque writes.")
            return 0
        if args.mode=="execute":
            print(f"Task: {source['instruction']}\nReference: {source['reference_image']}")
            print("Arrange bowl/coaster to match mapped_object_origins in replay_manifest.json.")
            answer=input("Full replay preflight passed. Type REPLAY to move the real arm, or ENTER to cancel: ").strip()
            if answer!="REPLAY":
                print("Cancelled; no replay trajectory sent.")
                return 0
        result=env._rpc(dict(op="replay_run",plan_id=prepared["plan_id"]))
        print(json.dumps(result))
        return 0 if result.get("status")=="replayed" else 2
    finally:
        env.close()


if __name__=="__main__":
    raise SystemExit(main())

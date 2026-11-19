"""Inspect/check SO-101, request a stop, or label a completed hardware evaluation.

Inspect and dry-run never write motor targets or torque. Check defaults to
offline mock devices. Motor execution is exposed only by the evaluation runner.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from inspect_robots.scene import Scene
from inspect_robots.types import Action
from inspect_robots_libero.hardware.embodiment import SO101HardwareEmbodiment
from inspect_robots_libero.evaluation_report import write_reports


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("inspect", "check"):
        action = sub.add_parser(name)
        action.add_argument("--config", type=Path)
        action.add_argument("--output", type=Path, required=True)
        if name == "check":
            action.add_argument("--mode", choices=("mock", "dry-run"), default="mock")
            action.add_argument("--delta-z", type=float, default=.005)
    stop = sub.add_parser("stop")
    stop.add_argument("--episode", type=Path, required=True, help="Exact episode hardware directory containing hardware_snapshot.json")
    score = sub.add_parser("score")
    score.add_argument("--evaluation", type=Path, required=True)
    score.add_argument("--run-id", required=True)
    score.add_argument("--success", choices=("yes", "no", "unknown"), required=True)
    score.add_argument("--note", default="")
    args = p.parse_args(argv)
    if args.command == "stop":
        if not (args.episode/"hardware_snapshot.json").is_file():
            p.error("Choose the exact hardware episode directory containing hardware_snapshot.json")
        (args.episode/"STOP").touch()
        print(f"Stop requested: {args.episode/'STOP'}")
        return
    if args.command == "score":
        path = args.evaluation/"evaluation.json"
        data = json.loads(path.read_text())
        rows = [r for r in data["runs"] if r["id"] == args.run_id and r["benchmark"] in ("hardware", "so101-upstream")]
        if len(rows) != 1:
            p.error("Select exactly one hardware run ID from evaluation.json")
        row = rows[0]
        if row["status"] in ("pending", "running", "not_run", "not_run_budget"):
            p.error("Human scoring requires a finished episode")
        if row.get("device_mode") != "execute" and args.success != "unknown":
            p.error("Mock/dry-run episodes cannot be labeled as physical task success/failure")
        label = {"yes": True, "no": False, "unknown": None}[args.success]
        audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), run_id=row["id"],
                     human_success=label, previous_human_success=row.get("human_success"), note=args.note)
        with (args.evaluation/"human_labels.jsonl").open("a") as file:
            file.write(json.dumps(audit)+"\n")
        row.update(human_success=label, success_source="human" if label is not None else "human_unknown", success_note=args.note)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data, indent=2))
        temp.replace(path)
        write_reports(args.evaluation)
        print(args.evaluation/"evaluation.xlsx")
        return
    mode = "inspect" if args.command == "inspect" else args.mode
    if mode != "mock" and args.config is None:
        p.error("Real-device inspection/dry-run needs an explicit --config")
    if args.command == "check" and (not math.isfinite(args.delta_z) or not 0 < args.delta_z <= .02):
        p.error("delta-z must be positive and at most .02 m")
    args.output.mkdir(parents=True, exist_ok=False)
    env = SO101HardwareEmbodiment(config_path=args.config, mode=mode, output_dir=args.output,
                                  max_steps=300, policy_seconds=120, auto_start=True, video=False)
    try:
        observation = env.reset(Scene(id="hardware_check", instruction="Offline/controller infrastructure check"))
        if args.command == "inspect":
            env._rpc({"op": "diagnostics"})
        if args.command == "check":
            target = observation.state["eef_state"].copy()
            target[2] += args.delta_z
            count = math.ceil(args.delta_z/env.info.action_space.semantics.max_step[2])+1
            path = np.linspace(observation.state["eef_state"], target, count)
            reason = env.preflight(path)
            if reason:
                raise RuntimeError(f"No motor commands sent; preflight rejected: {reason}")
            for point in path:
                result = env.step(Action(data=point))
                if result.terminated:
                    raise RuntimeError(f"Controller stopped: {result.termination_reason}")
            if mode == "dry-run":
                assert env.last["steps"] == env.last["motor_writes"] == 0
            else:
                assert abs(result.observation.state["eef_state"][2]-target[2]) < .005
        (args.output/"pose_check.json").write_text(json.dumps(env._last_state, indent=2))
    finally:
        env.close()
    report = dict(passed=True, real_api_calls=0, mode=mode, **(env.last or {}))
    (args.output/"check_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if args.command == "inspect":
        print(f"Servo settings and health: {args.output/'motor_diagnostics.json'}")


if __name__ == "__main__":
    main()

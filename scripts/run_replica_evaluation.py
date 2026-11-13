"""Compare low/medium/high reasoning with one runner for Replica and LIBERO.

Default: write an unexecuted plan and spreadsheet. --mock runs the real physics
and upstream policy against local HTTP replies. Only --execute-paid-rollouts
can connect to a provider, and it requires explicit models and USD caps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
import numpy as np
from inspect_robots import eval as evaluate
from inspect_robots._dotenv import read_dotenv
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots_agent.policy import agent_policy
from inspect_robots_libero.budget import BudgetTransport, PRICING, wire_for
from inspect_robots_libero.embodiment import LiberoEmbodiment, BASE_DOCS
from inspect_robots_libero.evaluation_report import write_reports
from inspect_robots_libero.replica.embodiment import ReplicaEmbodiment, replica_task, DOCS
from inspect_robots_libero.replica.scenes import read_scene, sha256
from inspect_robots_libero.tasks import libero_10_task, LIBERO_10_TASKS

ROOT = Path(__file__).resolve().parents[1]
EFFORTS = ("low", "medium", "high")
KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class AttemptRecorder(httpx.BaseTransport):
    """Count actual wire attempts, including retries and interrupted requests."""
    def __init__(self, inner, owner):
        self.inner, self.owner, self.attempts = inner, owner, 0

    def handle_request(self, request):
        self.attempts += 1
        self.owner.record(dict(event="http_attempt_started", call=self.owner.calls, attempt=self.attempts))
        try:
            return self.inner.handle_request(request)
        except BaseException:
            self.owner.uncertain = True
            raise

    def close(self):
        self.inner.close()


class RecordedLibero(LiberoEmbodiment):
    """Keep the LIBERO half identical to the existing cost-pilot adapter."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_hash, self.trajectory = None, []

    def _observation(self, data):
        obs = super()._observation(data)
        if self.initial_hash is None:
            digest = hashlib.sha256()
            for name in sorted(obs.images):
                digest.update(np.asarray(obs.images[name]).tobytes())
            digest.update(np.asarray(data["eef_state"]).tobytes())
            self.initial_hash = digest.hexdigest()
        self.trajectory.append(dict(step=data["steps"], eef=data["eef_state"]))
        return obs


def mock_transport(model, wire):
    """Only local replies. Exercises real tools, motion, images and accounting."""
    calls = 0

    def reply(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            name, arguments = "move_to", dict(targets={"gripper": 0.}, note="Offline check: open the gripper in place.")
        else:
            name, arguments = "give_up", dict(reason="Offline integration check complete.", hindsight="none")
        if wire == "responses":
            return httpx.Response(200, json=dict(id=f"mock_{calls}", status="completed",
                output=[dict(type="function_call", id=f"fc_{calls}", call_id=f"call_{calls}",
                             name=name, arguments=json.dumps(arguments))],
                usage=dict(input_tokens=100, input_tokens_details=dict(cached_tokens=20),
                           output_tokens=50, output_tokens_details=dict(reasoning_tokens=10))))
        return httpx.Response(200, json=dict(id=f"mock_{calls}", type="message", role="assistant",
            content=[dict(type="tool_use", id=f"tool_{calls}", name=name, input=arguments)],
            stop_reason="tool_use", stop_sequence=None,
            usage=dict(input_tokens=80, cache_creation_input_tokens=0,
                       cache_read_input_tokens=20, output_tokens=50)))
    return httpx.MockTransport(reply)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=("replica", "libero"), default="replica")
    p.add_argument("--catalog", type=Path)
    p.add_argument("--scene-json", type=Path, action="append")
    p.add_argument("--subset", choices=("ID", "OOD", "AB"), action="append")
    p.add_argument("--task-id", type=int, action="append", help="Replica repository task number 1..10; LIBERO native task number 0..9")
    p.add_argument("--variation", type=int, action="append", help="Replica reference pic/variant number 1..5")
    p.add_argument("--init-state", type=int, default=0, help="LIBERO initial state, 0..49")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--model", action="append", help="Repeat for multiple models; existing budget.py model options")
    p.add_argument("--effort", choices=EFFORTS, action="append", help="Default: low, medium, high")
    p.add_argument("--max-calls", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument("--total-budget", type=float)
    p.add_argument("--per-run-budget", type=float)
    p.add_argument("--pricing-file", type=Path, help="JSON mapping provider/model to four per-million input/cache-read/cache-write/output rates")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-video", action="store_true", help="Replica only")
    p.add_argument("--viewer", action="store_true", help="Replica: watch live simulation in a desktop window at 20 Hz")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--mock", action="store_true")
    modes.add_argument("--execute-paid-rollouts", action="store_true")
    modes.add_argument("--report-only", action="store_true")
    return p


def select_scenes(args):
    if args.benchmark == "libero":
        if args.catalog or args.scene_json or args.subset or args.variation:
            raise ValueError("Replica scene selectors cannot be used with LIBERO")
        if not 0 <= args.init_state < 50:
            raise ValueError("LIBERO init state must be in 0..49")
        ids = args.task_id or [0]
        if any(i not in range(10) for i in ids):
            raise ValueError("LIBERO task IDs must be in 0..9")
        return [dict(id=f"libero_task{i:02d}_init{args.init_state:02d}", task_id=i,
                     instruction=LIBERO_10_TASKS[i][1], subset="libero_10", variation=args.init_state,
                     fidelity="libero_native", calibration_status="native_fixed_initial_state") for i in ids]
    paths = list(args.scene_json or [])
    if args.catalog:
        catalog = json.loads(args.catalog.read_text())
        paths += [args.catalog.resolve().parent/entry["config"] for entry in catalog["scenes"]]
    if not paths:
        raise ValueError("Replica requires --catalog or --scene-json")
    scenes = []
    seen = set()
    for path in paths:
        path = Path(path).resolve()
        scene = read_scene(path)
        if args.subset and scene["subset"] not in args.subset:
            continue
        if args.task_id and scene["task_id"] not in args.task_id:
            continue
        if args.variation and scene["variation"] not in args.variation:
            continue
        if scene["id"] in seen:
            raise ValueError(f"Duplicate scene ID {scene['id']}")
        if not path.with_suffix(".xml").is_file():
            raise ValueError(f"Compile scene before running: {path}")
        seen.add(scene["id"])
        scene.update(scene_config=str(path), scene_json_sha256=sha256(path),
                     scene_xml_sha256=sha256(path.with_suffix(".xml")),
                     calibration_status=scene["calibration"]["status"])
        scenes.append(scene)
    if not scenes:
        raise ValueError("No scenes matched the selectors")
    return scenes


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.viewer and args.benchmark != "replica":
        p.error("--viewer is currently supported for Replica only")
    out = args.output.resolve()
    if args.report_only:
        write_reports(out)
        print(out/"evaluation.xlsx")
        return
    for name in ("max_calls", "max_steps", "repeat", "max_output_tokens"):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if args.max_output_tokens > 128000:
        p.error("max-output-tokens must be <=128000")
    execute = args.mock or args.execute_paid_rollouts
    if execute and not args.model:
        p.error("Choose at least one --model before execution")
    if args.execute_paid_rollouts and (args.total_budget is None or args.per_run_budget is None):
        p.error("Paid execution requires --total-budget and --per-run-budget")
    for value in (args.total_budget, args.per_run_budget):
        if value is not None and (not math.isfinite(value) or value <= 0):
            p.error("Budgets must be positive and finite")
    try:
        if args.pricing_file:
            pricing = json.loads(args.pricing_file.read_text())
            for model, rates in pricing.items():
                if model.partition("/")[0] not in KEYS or not isinstance(rates, list) or len(rates) != 4 or any(
                        not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in rates):
                    raise ValueError(f"Invalid price row {model}")
                if rates[2] < max(rates[0], rates[1]):
                    raise ValueError(f"Conservative cache-write reservation rate must cover all input rates: {model}")
                PRICING[model] = tuple(rates)
        models = args.model or ["awaiting-model-selection"]
        if args.model and any(model not in PRICING for model in models):
            raise ValueError("Unknown model pricing; use an existing budget.py model or provide --pricing-file")
        efforts = args.effort or list(EFFORTS)
        if len(models) != len(set(models)) or len(efforts) != len(set(efforts)):
            raise ValueError("Model and effort selections must be unique")
        scenes = select_scenes(args)
    except (ValueError, KeyError, OSError) as exc:
        p.error(str(exc))
    settings = {**read_dotenv(ROOT/".env"), **os.environ} if args.execute_paid_rollouts else {}
    if args.execute_paid_rollouts:
        for model in models:
            if not settings.get(KEYS[model.partition("/")[0]]):
                p.error(f"Missing {KEYS[model.partition('/')[0]]}")
    mode = "mock" if args.mock else "paid" if args.execute_paid_rollouts else "planned"
    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), benchmark=args.benchmark,
        accounting_mode=mode, models=models, efforts=efforts, max_calls=args.max_calls,
        max_steps=args.max_steps, max_output_tokens=args.max_output_tokens, control_hz=20,
        image_horizon=2, eval_seed=42, total_budget_usd=(args.total_budget or 100.) if args.mock else args.total_budget,
        per_run_budget_usd=(args.per_run_budget or 5.) if args.mock else args.per_run_budget, zero_shot=True,
        live_viewer=args.viewer, motion_wall_pacing=args.viewer,
        pricing_per_million={m: PRICING[m] for m in models if m in PRICING},
        pricing_source=str(args.pricing_file.resolve()) if args.pricing_file else "inherited repository budget.py snapshot; not reverified by this runner",
        pricing_file_sha256=sha256(args.pricing_file) if args.pricing_file else None,
        project_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
        inspect_robots_commit=(ROOT/"upstream_commit.txt").read_text().strip(),
        embodiment_prompt=DOCS if args.benchmark == "replica" else BASE_DOCS,
        scene_snapshots=scenes,
        notes=["Replica task IDs follow task filenames, not paper Table 3 numbering.",
               "Layouts marked approximate have not been calibrated to reference photographs.",
               "Mock costs are simulated and do not represent API spending.",
               "Total agent cost includes all API tokens; reasoning output cost is an optional subset."])
    protocol["embodiment_prompt_sha256"] = hashlib.sha256(protocol["embodiment_prompt"].encode()).hexdigest()
    runs = []
    for scene in scenes:
        for repeat in range(1, args.repeat+1):
            for mi, model in enumerate(models):
                for effort in efforts:
                    ident = f"{scene['id']}_r{repeat:02d}_m{mi:02d}_{effort}"
                    runs.append(dict(id=ident, run_folder=ident, benchmark=args.benchmark,
                        subset=scene["subset"], task_id=scene["task_id"], variation=scene["variation"],
                        repeat=repeat, instruction=scene["instruction"], model=model, effort=effort,
                        accounting_mode=mode, status="pending", fidelity=scene["fidelity"],
                        calibration_status=scene["calibration_status"], scene_id=scene["id"], steps=None))
    data = dict(protocol=protocol, status="running" if execute else "planned", budget_charge_usd=0., runs=runs)

    def save():
        temp = out/"evaluation.json.tmp"
        temp.write_text(json.dumps(data, indent=2))
        temp.replace(out/"evaluation.json")
        write_reports(out)

    save()
    if not execute:
        print(f"Planned {len(runs)} episodes; no API or simulator calls. Spreadsheet: {out/'evaluation.xlsx'}")
        return
    total_budget = args.total_budget or 100.
    per_budget = args.per_run_budget or 5.
    try:
        for row in runs:
            remaining = total_budget-data["budget_charge_usd"]
            if remaining <= 0:
                row["status"] = "not_run_budget"
                continue
            scene = next(s for s in scenes if s["id"] == row["scene_id"])
            folder = out/row["run_folder"]
            wire = wire_for(row["model"])
            inner = mock_transport(row["model"], wire) if args.mock else httpx.HTTPTransport(retries=0)
            cap = min(remaining, per_budget)
            transport = BudgetTransport(folder/"api", cap, cap, inner=inner, model=row["model"],
                task_id=scene["task_id"], max_calls=args.max_calls, image_horizon=2,
                max_output_tokens=args.max_output_tokens)
            transport.inner = AttemptRecorder(inner, transport)
            transport.begin_episode(0)
            env = policy = None
            row["status"] = "running"
            start = time.monotonic()
            save()
            try:
                if args.benchmark == "replica":
                    env = ReplicaEmbodiment(scene_config=scene["scene_config"], output_dir=folder/"sim",
                                            max_steps=args.max_steps, video=not args.no_video, viewer=args.viewer)
                    task = replica_task(scene["scene_config"], max_steps=args.max_steps)
                else:
                    env = RecordedLibero(task_id=scene["task_id"], output_dir=folder/"sim",
                                         libero_python=settings.get("LIBERO_PYTHON"))
                    task = libero_10_task(scene["task_id"], start=args.init_state, max_steps=args.max_steps)
                key_name = KEYS[row["model"].partition("/")[0]]
                policy = agent_policy(model=row["model"], env={key_name: "offline-fake" if args.mock else settings[key_name]},
                    wire=wire, effort=row["effort"], max_llm_calls=args.max_calls, image_horizon=2,
                    prior_learnings=None, transport=transport, wire_capture=False)
                space = env.info.action_space
                (log,) = evaluate(task, policy, env, seed=42, log_dir=str(folder/"inspect"),
                    store_frames=True, store_actions=True, grader=None,
                    approver=ChainApprover(ClampApprover(space), DeltaLimitApprover(space)))
                row.update(status="finished" if log.status == "success" else "error", framework_status=log.status,
                           termination_reasons=[reason for sample in log.samples for reason in sample.termination_reasons])
            except KeyboardInterrupt:
                row["status"] = "interrupted"
                raise
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__)
            finally:
                # Preserve accounting even when initialization or simulator cleanup fails.
                try:
                    if env is not None:
                        env.close()
                finally:
                    try:
                        if policy is not None:
                            policy._client.close()
                    finally:
                        transport.dispose()
                        row.update(wall_seconds=time.monotonic()-start, usage_unknown=transport.uncertain,
                                   budget_charge_usd=transport.used, stop_detail=transport.blocked)
                        if env is not None:
                            last = env.last or {}
                            row.update(steps=last.get("steps", 0), environment_success=bool(last.get("success", False)),
                                       initial_observation_sha256=env.initial_hash)
                            (folder/"trajectory.json").write_text(json.dumps(env.trajectory))
                        data["budget_charge_usd"] += transport.used
                        if transport.uncertain:
                            row["status"] = "usage_unknown"
                        elif transport.blocked:
                            row["status"] = "budget_stopped" if "budget" in transport.blocked else "stopped"
                        elif row.get("environment_success"):
                            row["status"] = "task_success"
                        save()
            print(json.dumps({k: row.get(k) for k in ("id", "status", "steps", "budget_charge_usd")}), flush=True)
            if transport.uncertain:
                data["status"] = "stopped_usage_unknown"
                break
        else:
            data["status"] = "completed"
    finally:
        if data["status"] == "running":
            data["status"] = "interrupted"
        for row in runs:
            if row["status"] == "pending":
                row["status"] = "not_run"
        save()
    print(f"Spreadsheet: {out/'evaluation.xlsx'}", flush=True)


if __name__ == "__main__":
    main()

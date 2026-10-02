"""Paid LLM rollouts on one LIBERO-10 task with hard spending caps.

Same Task / Embodiment / agent policy as ``inspect-robots run``, plus:
  * a fail-closed per-episode and total USD budget (``BudgetTransport``),
  * a fresh policy per episode (no cross-episode memory),
  * optional demonstration storyboards (few-shot) and task notes,
  * a ``protocol.json`` + ``summary.json`` recording exactly what ran.

Keys are read from ``.env`` in the repo root (or the environment) and are only
handed to the policy; they are never written to disk or passed to LIBERO.
Nothing is sent unless ``--execute-paid-rollouts`` is given.

Example (one zero-shot episode of native task 2, at most $3):
    python scripts/run_budgeted_rollouts.py --model openai/gpt-6-sol --task-id 2 \
        --episodes 1 --total-budget 3 --output outputs/task2_sol_zeroshot \
        --execute-paid-rollouts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from inspect_robots import eval as evaluate
from inspect_robots._dotenv import read_dotenv
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.task import Task
from inspect_robots_agent.policy import agent_policy

from inspect_robots_libero.budget import PRICING, BudgetTransport, wire_for
from inspect_robots_libero.embodiment import LiberoEmbodiment
from inspect_robots_libero.tasks import LIBERO_10_TASKS, NUM_INIT_STATES, libero_10_task

ROOT = Path(__file__).resolve().parents[1]
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def load_settings() -> dict[str, str]:
    """Environment variables win over ``.env`` (same precedence as the upstream CLI)."""
    return {**read_dotenv(ROOT / ".env"), **os.environ}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(PRICING), required=True)
    parser.add_argument("--task-id", type=int, choices=range(len(LIBERO_10_TASKS)), required=True,
                        help="native LIBERO-10 task id (see README)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="episodes on init states start..start+N-1")
    parser.add_argument("--start", type=int, default=0, help="first LIBERO init state")
    parser.add_argument("--total-budget", type=float, required=True, help="USD cap for the run")
    parser.add_argument("--episode-budget", type=float, default=3.0, help="USD cap per episode")
    parser.add_argument("--max-calls", type=int, default=20, help="LLM requests per episode")
    parser.add_argument("--effort", choices=EFFORTS, default="low")
    parser.add_argument("--image-horizon", type=int, default=2,
                        help="how many past observations keep their images")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--max-output-tokens", type=int, default=2048,
                        help="output-token cap per OpenAI request, reasoning included")
    parser.add_argument("--demo-manifest", type=Path,
                        help="storyboards from scripts/prepare_demonstrations.py (few-shot)")
    parser.add_argument("--task-notes", action="store_true",
                        help="append demonstration-derived hints (not zero-shot)")
    parser.add_argument("--output", type=Path, required=True, help="new directory for this run")
    parser.add_argument("--execute-paid-rollouts", action="store_true",
                        help="required: acknowledges that this sends paid API requests")
    args = parser.parse_args()
    if not args.execute_paid_rollouts:
        parser.error("paid requests require --execute-paid-rollouts")
    if args.total_budget <= 0 or args.episode_budget <= 0:
        parser.error("budgets must be positive")
    if not 0 <= args.start < args.start + args.episodes <= NUM_INIT_STATES:
        parser.error(f"init states must stay within 0..{NUM_INIT_STATES - 1}")

    settings = load_settings()
    provider = args.model.partition("/")[0]
    key_env = KEY_ENV[provider]
    if not settings.get(key_env):
        parser.error(f"{key_env} is not set; put it in {ROOT / '.env'} (see .env.example)")

    # A fresh directory per run: never resume a paid run and risk double billing.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.umask(0o077)
    transport = BudgetTransport(
        output / "api", args.total_budget, args.episode_budget, task_id=args.task_id,
        demo_manifest=args.demo_manifest, max_calls=args.max_calls, model=args.model,
        image_horizon=args.image_horizon, max_output_tokens=args.max_output_tokens,
    )
    full_task = libero_10_task(args.task_id, num_scenes=args.episodes, start=args.start,
                               max_steps=args.max_steps)
    protocol = {
        "model": args.model,
        "wire": wire_for(args.model),
        "task_id": args.task_id,
        "task_name": LIBERO_10_TASKS[args.task_id][0],
        "instruction": LIBERO_10_TASKS[args.task_id][1],
        "episodes": args.episodes,
        "initial_state_ids": [s.metadata["init_state"] for s in full_task.scenes],
        "env_seeds": [s.init_seed for s in full_task.scenes],
        "max_steps": args.max_steps,
        "max_api_attempts_per_episode": args.max_calls,
        "reasoning_effort": args.effort,
        "image_horizon": args.image_horizon,
        "max_output_tokens": args.max_output_tokens,
        "total_budget": args.total_budget,
        "episode_budget": args.episode_budget,
        "zero_shot": args.demo_manifest is None and not args.task_notes,
        "task_notes": args.task_notes,
        "demonstrations": json.loads(args.demo_manifest.read_text()) if args.demo_manifest else False,
        "prior_learnings": None,
        "upstream_commit": (ROOT / "upstream_commit.txt").read_text().strip(),
        "budget_note": "Conservative local reservations; provider billing remains authoritative.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    results: list[dict[str, object]] = []

    def write_summary() -> None:
        (output / "summary.json").write_text(json.dumps(
            {"protocol": protocol, "results": results,
             "total_estimated_usd": transport.estimated,
             "total_budget_charge_usd": transport.used},
            indent=2))

    try:
        for ep, scene in enumerate(full_task.scenes):
            if transport.uncertain or transport.used >= args.total_budget:
                results.append({"episode": ep, "scene": scene.id,
                                "status": "not_run_budget_or_usage_unknown"})
                continue
            transport.begin_episode(ep)
            print(f"[episode {ep}] {scene.id}: starting simulator "
                  f"(up to {args.max_calls} LLM calls, {args.max_steps} steps)", file=sys.stderr, flush=True)
            folder = output / f"episode_{ep:02d}"
            embodiment = LiberoEmbodiment(task_id=args.task_id, output_dir=folder,
                                          libero_python=settings.get("LIBERO_PYTHON"),
                                          task_notes=args.task_notes)
            # No base_url: upstream resolves the provider from the model prefix.
            policy = agent_policy(
                model=args.model, env={key_env: settings[key_env]}, wire=wire_for(args.model),
                effort=args.effort, max_llm_calls=args.max_calls,
                image_horizon=args.image_horizon, prior_learnings=None,
                transport=transport, wire_capture=True,
            )
            task = Task(name=full_task.name, scenes=[scene], scorer=full_task.scorer,
                        max_steps=args.max_steps)
            # The CLI's default guardrails (bounds clamp + per-step delta limit);
            # the raw eval() API applies none unless asked.
            space = embodiment.info.action_space
            approver = ChainApprover(ClampApprover(space), DeltaLimitApprover(space))
            try:
                (log,) = evaluate(task, policy, embodiment, log_dir=str(folder / "inspect"),
                                  seed=42 + ep, store_frames=True, store_actions=True,
                                  grader=None, approver=approver)
                last = embodiment.last or {}
                success = bool(last.get("success"))
                if success:
                    status = "success"
                elif transport.uncertain:
                    status = "api_error_usage_unknown"
                elif transport.blocked or transport.calls >= args.max_calls:
                    status = "incomplete_budget"
                else:
                    status = "error" if log.status == "error" else "failure"
                row = {"episode": ep, "scene": scene.id, "status": status,
                       "environment_success": success, "framework_status": log.status,
                       "metrics": log.results.metrics if log.results else None,
                       "steps": last.get("steps", 0), "api_attempts": transport.calls,
                       "estimated_usd": transport.episode_estimated,
                       "budget_charge_usd": transport.episode_used,
                       "stop_detail": transport.blocked,
                       "video": str(embodiment.episode_dir / "rollout.mp4")
                       if embodiment.episode_dir else None}
            finally:
                embodiment.close()
                policy._client.close()
            results.append(row)
            write_summary()
            print(f"[episode {ep}] finished: {row.get('status')}, {row.get('steps')} steps, "
                  f"{row.get('api_attempts')} LLM calls", file=sys.stderr, flush=True)
            print(json.dumps(row), flush=True)
    finally:
        transport.dispose()
        write_summary()


if __name__ == "__main__":
    main()

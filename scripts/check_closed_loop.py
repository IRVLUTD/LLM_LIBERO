"""Offline closed loop: real agent policy + eval + LIBERO, with the LLM API mocked.

Exercises the full stack (tool calls -> interpolation -> worker -> LIBERO ->
scoring -> budget accounting) for one provider wire without spending anything.
No key is needed; the fake key never leaves this process.

    python scripts/check_closed_loop.py --model openai/gpt-6-astra
    python scripts/check_closed_loop.py --model anthropic/claude-opus-5-5 --task-id 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx
from inspect_robots import eval as evaluate
from inspect_robots._dotenv import read_dotenv
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots_agent.policy import agent_policy

from inspect_robots_libero.budget import PRICING, BudgetTransport, wire_for
from inspect_robots_libero.embodiment import LiberoEmbodiment
from inspect_robots_libero.tasks import libero_10_task

ROOT = Path(__file__).resolve().parents[1]
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", choices=sorted(PRICING), default="openai/gpt-6-astra")
parser.add_argument("--task-id", type=int, default=0)
parser.add_argument("--demo-manifest", type=Path)
args = parser.parse_args()
wire = wire_for(args.model)
out = ROOT / "outputs" / "checks" / f"closed_loop_{wire}_task{args.task_id:02d}"
out.mkdir(parents=True, exist_ok=True)

calls: list[dict[str, Any]] = []


def fake(request: httpx.Request) -> httpx.Response:
    """Turn 1: open the gripper in place. Turn 2: give up."""
    calls.append(json.loads(request.content))
    idx = len(calls)
    if idx == 1:
        name, tool_args = "move_to", {"targets": {"gripper": 0.0},
                                      "note": "Offline mock: hold pose and open gripper."}
    else:
        name, tool_args = "give_up", {"reason": "Offline test completed", "hindsight": "none"}
    if wire == "responses":
        return httpx.Response(200, headers={"x-request-id": f"offline-{idx}"}, json={
            "id": f"offline-{idx}", "status": "completed",
            "output": [{"type": "function_call", "id": f"fc_{idx}", "call_id": f"call_{idx}",
                        "name": name, "arguments": json.dumps(tool_args)}],
            "usage": {"input_tokens": 100, "output_tokens": 100}})
    return httpx.Response(200, headers={"request-id": f"offline-{idx}"}, json={
        "id": f"msg_offline_{idx}", "type": "message", "role": "assistant",
        "content": [{"type": "tool_use", "id": f"toolu_{idx}", "name": name, "input": tool_args}],
        "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 50,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}})


settings = {**read_dotenv(ROOT / ".env"), **os.environ}
key_env = KEY_ENV[args.model.partition("/")[0]]
transport = BudgetTransport(out / "mock_accounting", 15, 3, inner=httpx.MockTransport(fake),
                            task_id=args.task_id, demo_manifest=args.demo_manifest,
                            model=args.model)
transport.begin_episode(0)
policy = agent_policy(model=args.model, env={key_env: "offline-fake-not-a-real-key"},
                      wire=wire, effort="low", max_llm_calls=3, transport=transport,
                      wire_capture=False)
env = LiberoEmbodiment(output_dir=out, libero_python=settings.get("LIBERO_PYTHON"))
approver = ChainApprover(ClampApprover(env.info.action_space),
                         DeltaLimitApprover(env.info.action_space))
try:
    (log,) = evaluate(libero_10_task(args.task_id, max_steps=60), policy, env,
                      log_dir=str(out / "inspect"), store_frames=True, approver=approver)
    assert log.status == "success", log.status
    assert len(calls) == 2, len(calls)
    assert log.results is not None and log.results.metrics["success_at_end"] == 0.0
    report = {"passed": True, "model": args.model, "wire": wire, "real_api_calls": 0,
              "mocked_requests": len(calls), "framework_status": log.status,
              "environment_success": bool(env.last and env.last["success"])}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
finally:
    env.close()
    policy._client.close()
    transport.dispose()

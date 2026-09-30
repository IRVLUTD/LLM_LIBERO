"""Nine full LIBERO episodes sharing one USD allowance; regenerates an offline HTML report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import subprocess

import httpx
import numpy as np
from inspect_robots import eval as evaluate
from inspect_robots._dotenv import read_dotenv
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots_agent.policy import agent_policy
from inspect_robots_libero.budget import BudgetTransport, PRICING
from inspect_robots_libero.embodiment import BASE_DOCS, LiberoEmbodiment
from inspect_robots_libero.tasks import libero_10_task, LIBERO_10_TASKS
from inspect_robots_libero.pilot_report import write_report

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['openai/gpt-6-luna', 'openai/gpt-6-sol', 'openai/gpt-6-astra']
EXTRA_MODELS = ['openai/gpt-6.1-sol']
EFFORTS = ['low', 'medium', 'high']
MODEL_LABELS = {
    'openai/gpt-6-luna': 'luna',
    'openai/gpt-6-sol': 'sol',
    'openai/gpt-6-astra': 'astra',
    'openai/gpt-6.1-sol': 'gpt_6_1_sol',
}
CONFIGURATIONS = [
    {'id': MODEL_LABELS[m] + '_' + e, 'model': m, 'effort': e}
    for m in MODELS + EXTRA_MODELS for e in EFFORTS
]


class RecordedEmbodiment(LiberoEmbodiment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trajectory = []
        self.initial_hash = None

    def _observation(self, data):
        obs = super()._observation(data)
        self.trajectory.append({'step': data['steps'], 'eef': data['eef_state'],
                                'gripper_qpos': data['gripper_qpos']})
        if self.initial_hash is None:
            digest = hashlib.sha256()
            for name in sorted(obs.images):
                digest.update(np.asarray(obs.images[name]).tobytes())
            digest.update(np.asarray(data['eef_state']).tobytes())
            self.initial_hash = digest.hexdigest()
        return obs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--total-budget', type=float, default=45,
                   help='hard cap across the complete sweep (default: 45)')
    p.add_argument('--per-run-budget', type=float, default=5,
                   help='hard cap for each model/effort episode (default: 5)')
    p.add_argument('--task-id', type=int, choices=range(10), default=0)
    p.add_argument('--init-state', type=int, choices=range(50), default=0)
    p.add_argument('--max-calls', type=int, default=100,
                   help='emergency per-episode LLM-call ceiling (default: 100)')
    p.add_argument('--max-steps', type=int, default=900,
                   help='simulator-step horizon per episode (default: 900)')
    p.add_argument('--max-output-tokens', type=int, default=8192)
    p.add_argument('--config', action='append', choices=[c['id'] for c in CONFIGURATIONS],
                   help='configuration to run; repeat to select and order runs')
    p.add_argument('--execute-paid-rollouts', action='store_true')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if args.report_only:
        write_report(args.output.resolve())
        return
    if not args.execute_paid_rollouts:
        p.error('paid requests require --execute-paid-rollouts')
    if (not math.isfinite(args.total_budget) or args.total_budget <= 0
            or not math.isfinite(args.per_run_budget) or args.per_run_budget <= 0
            or args.max_calls < 1 or args.max_steps < 1):
        p.error('budgets, call ceiling, and step horizon must be positive and finite')
    if not 1 <= args.max_output_tokens <= 128000:
        p.error('max output tokens must be in 1..128000')
    selected_ids = args.config or [c['id'] for c in CONFIGURATIONS if c['model'] in MODELS]
    if len(selected_ids) != len(set(selected_ids)):
        p.error('--config values must be unique')
    selected = [next(c for c in CONFIGURATIONS if c['id'] == ident) for ident in selected_ids]
    selected_models = list(dict.fromkeys(c['model'] for c in selected))
    settings = {**read_dotenv(ROOT / '.env'), **os.environ}
    if not settings.get('OPENAI_API_KEY'):
        p.error('OPENAI_API_KEY is missing from .env and the environment')
    os.umask(0o077)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'task_id': args.task_id, 'instruction': LIBERO_10_TASKS[args.task_id][1],
        'init_state': args.init_state, 'env_seed': 142 + args.init_state, 'eval_seed': 42,
        'max_calls': args.max_calls, 'max_steps': args.max_steps, 'image_horizon': 2,
        'max_output_tokens': args.max_output_tokens, 'total_budget_usd': args.total_budget,
        'per_run_budget_usd': args.per_run_budget,
        'zero_shot': True, 'service_tier': 'default', 'wire': 'responses',
        'python': platform.python_version(),
        'project_commit': subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,
                                         capture_output=True,text=True).stdout.strip(),
        'inspect_robots_commit': (ROOT/'upstream_commit.txt').read_text().strip(),
        'embodiment_prompt_sha256': hashlib.sha256(BASE_DOCS.encode()).hexdigest(),
        'embodiment_prompt': BASE_DOCS,
        'pricing_per_million': {m: PRICING[m] for m in selected_models},
        'pricing_fields': ['input','cached_input','cache_write','output'],
        'pricing_verified_date': '2026-09-29',
        'pricing_sources': ['https://developers.openai.com/api/docs/models/'+m.split('/')[1] for m in selected_models],
        'notes': ['One full episode per configuration on one task and initial state; no success-rate inference.',
                  f'Model is told the {args.max_calls}-call emergency ceiling; success, give-up, or the {args.max_steps}-step simulator horizon can end the episode first.',
                  'Selected configurations run sequentially in the recorded order. Cache hits are not controlled.',
                  f'Each configuration has its own ${args.per_run_budget:g} cap within the ${args.total_budget:g} sweep cap.',
                  'Full output limit reserved before each call; input reservation remains a conservative estimate.',
                  'Unknown usage stops the entire pilot. Provider billing remains authoritative.',
                  'Costs exclude local GPU time and electricity.']}
    data = {'protocol': protocol, 'status': 'running', 'estimated_usd': 0.,
            'budget_charge_usd': 0., 'runs': [dict(c, status='pending') for c in selected]}
    def save():
        tmp = out/'pilot.json.tmp'
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(out/'pilot.json')
        write_report(out)
    save()
    try:
        for row in data['runs']:
            remaining = args.total_budget - data['budget_charge_usd']
            if remaining <= 0:
                row['status'] = 'not_run_budget'
                continue
            folder = out/row['id']
            run_budget = min(args.per_run_budget, remaining)
            transport = BudgetTransport(folder/'api', run_budget, run_budget,
                inner=httpx.HTTPTransport(retries=0), task_id=args.task_id,
                max_calls=args.max_calls, model=row['model'], image_horizon=2,
                max_output_tokens=args.max_output_tokens)
            transport.begin_episode(0)
            env = RecordedEmbodiment(task_id=args.task_id, output_dir=folder/'sim',
                                      libero_python=settings.get('LIBERO_PYTHON'))
            policy = None
            start = time.monotonic()
            row['status'] = 'running'
            save()
            print(f"Starting {row['id']}; run cap ${run_budget:.2f}, sweep remaining ${remaining:.4f}", flush=True)
            try:
                policy = agent_policy(model=row['model'], env={'OPENAI_API_KEY': settings['OPENAI_API_KEY']},
                    wire='responses', effort=row['effort'], max_llm_calls=args.max_calls,
                    image_horizon=2,
                    prior_learnings=None, transport=transport, wire_capture=False)
                space = env.info.action_space
                (log,) = evaluate(libero_10_task(args.task_id, start=args.init_state, max_steps=args.max_steps),
                    policy, env, log_dir=str(folder/'inspect'), seed=42, store_frames=True,
                    store_actions=True, grader=None,
                    approver=ChainApprover(ClampApprover(space), DeltaLimitApprover(space)))
                row['framework_status'] = log.status
                row['status'] = 'finished'
            except Exception as exc:
                # Do not persist exception messages that could contain credentials.
                row['status'] = 'error'
                row['error_type'] = type(exc).__name__
            finally:
                try:
                    env.close()
                finally:
                    if policy is not None:
                        policy._client.close()
                    transport.dispose()
                    row.update(wall_seconds=time.monotonic()-start, api_attempts=transport.calls,
                        estimated_usd=transport.estimated, budget_charge_usd=transport.used,
                        usage_unknown=transport.uncertain, stop_detail=transport.blocked,
                        steps=(env.last or {}).get('steps',0),
                        environment_success=bool((env.last or {}).get('success',False)),
                        initial_observation_sha256=env.initial_hash,
                        trajectory=env.trajectory,
                        video=str((env.episode_dir/'rollout.mp4').relative_to(out)) if env.episode_dir else None)
                    data['estimated_usd'] += transport.estimated
                    data['budget_charge_usd'] += transport.used
                    if transport.uncertain:
                        row['status'] = 'usage_unknown'
                    elif transport.blocked:
                        row['status'] = 'budget_stopped' if 'budget' in transport.blocked else 'stopped'
                    elif row['environment_success']:
                        row['status'] = 'task_success'
                    save()
            print(json.dumps({k: v for k,v in row.items() if k not in ('trajectory',)}), flush=True)
            if transport.uncertain:
                data['status'] = 'stopped_usage_unknown'
                break
        else:
            data['status'] = 'completed'
    finally:
        if data['status'] == 'running':
            data['status'] = 'interrupted'
        for row in data['runs']:
            if row['status'] == 'pending':
                row['status'] = 'not_run'
        save()
    print(f"Report: {out/'report.html'}", flush=True)


if __name__ == '__main__':
    main()

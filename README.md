# LLM_LIBERO

Run frontier LLMs (GPT, Claude, Gemini, anything behind an OpenAI-compatible
API) as closed-loop control policies on the [LIBERO-10](https://libero-project.github.io/)
manipulation benchmark.

This repo is a plugin for [Inspect Robots](https://github.com/robocurve/inspect-robots),
built the same way as the plugins in that repo (entry points, `EmbodimentBase`,
`Task`, `-E`/`-T` arguments). It adds two registry entries:

| Registry kind | Name | What it is |
|---|---|---|
| embodiment | `libero` | A LIBERO Panda arm in MuJoCo, 20 Hz, absolute end-effector pose actions, agentview + wrist RGB |
| task | `libero_10` | One LIBERO-10 task, one scene per LIBERO initial state, scored by LIBERO's own success predicate |

The LLM side is the upstream `agent` policy from `inspect-robots-agent`,
used unchanged. The model sees the camera images and robot state, calls tools
such as `move_to`, and the upstream framework turns each call into actions.

This repo also adds a runner for paid experiments with hard USD spending
caps, and optional few-shot demonstration storyboards.

## Contents

1. [Installation](#installation)
2. [API keys](#api-keys)
3. [Check the installation](#check-the-installation)
4. [LIBERO-10 tasks](#libero-10-tasks)
5. [Run one LIBERO-10 rollout](#run-one-libero-10-rollout)
6. [Use a different LLM](#use-a-different-llm)
7. [Paid runs with spending caps](#paid-runs-with-spending-caps)
8. [Few-shot demonstrations](#few-shot-demonstrations)
9. [Python API](#python-api)
10. [Implementation notes](#implementation-notes)
11. [Troubleshooting](#troubleshooting)

## Installation

Tested on Linux with an NVIDIA GPU (headless EGL rendering). You need `git`
and `conda`. Everything, LIBERO and the LLM agent stack, goes into one
Python 3.12 environment.

### 1. Clone this repo and Inspect Robots

Inspect Robots is installed from source at a pinned commit (`upstream_commit.txt`):

```bash
git clone git@github.com:IRVLUTD/LLM_LIBERO.git
cd LLM_LIBERO
git clone https://github.com/robocurve/inspect-robots.git vendor/inspect-robots
git -C vendor/inspect-robots checkout "$(cat upstream_commit.txt)"
```

### 2. Create the environment

```bash
conda create -n llm-libero python=3.12 -y
conda activate llm-libero
pip install hf-libero==0.1.4 imageio imageio-ffmpeg h5py
pip install -r requirements.lock.txt \
  ./vendor/inspect-robots \
  ./vendor/inspect-robots/plugins/inspect-robots-agent
pip install -e .
```

`hf-libero` is Hugging Face's packaging of LIBERO. It installs robosuite 1.4.0,
MuJoCo 3.x and bddl 1.0.1, and downloads the scene assets (about 400 MB) from
the Hugging Face Hub the first time a scene loads.

### 3. Initialise LIBERO once

The first import asks
`Do you want to specify a custom path for the dataset folder? (Y/N)`. Answer `N`:

```bash
python -c "import libero.libero"
```

This writes `~/.libero/config.yaml`. The simulator refuses to start without it,
since the question would otherwise block it.

### 4. Configure `.env`

```bash
cp .env.example .env
chmod 600 .env
```

Add your API keys (next section).

All commands below assume the `llm-libero` environment is active.

## API keys

Keys live only in `.env` in the repo root. `.gitignore` excludes it
(and any `.env.*` backup), and only `.env.example` is tracked:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
# GEMINI_API_KEY=...
```

Fill in only the providers you use. The variable names are the ones
`inspect-robots-agent` looks for (see [Use a different LLM](#use-a-different-llm)).

Run commands from the repo root so `.env` is found. Variables already set in
your shell take precedence.

> [!WARNING]
> Never commit `.env`. Before your first push, run
> `git status --ignored` to confirm `.env` is listed as ignored. If a key is
> ever committed, revoke it at the provider. Removing it from git history
> is not enough.

## Check the installation

None of these checks calls a real LLM or needs an API key.

```bash
# 1. Unit tests: task factory, registry wiring, budget accounting (mocked HTTP)
python -m unittest discover -s tests -v

# 2. The plugin is discovered by the upstream registry
inspect-robots list embodiments   # includes: libero
inspect-robots list tasks         # includes: libero_10

# 3. Simulator only: lift 3 cm, close and open the gripper, check LIBERO tracked it
python scripts/check_adapter.py

# 4. Full closed loop with a mocked LLM (one run per provider wire)
python scripts/check_closed_loop.py --model openai/gpt-6-astra
python scripts/check_closed_loop.py --model anthropic/claude-opus-5-5
```

Checks 3 and 4 each print a JSON line with `"passed": true`. Their outputs go
to `outputs/checks/`.

## LIBERO-10 tasks

`task_id` is LIBERO's native order (`benchmark.get_benchmark_dict()["libero_10"]()`):

| `task_id` | Scene | Instruction |
|---|---|---|
| 0 | LIVING_ROOM_SCENE2 | put both the alphabet soup and the tomato sauce in the basket |
| 1 | LIVING_ROOM_SCENE2 | put both the cream cheese box and the butter in the basket |
| 2 | KITCHEN_SCENE3 | turn on the stove and put the moka pot on it |
| 3 | KITCHEN_SCENE4 | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | LIVING_ROOM_SCENE5 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | STUDY_SCENE1 | pick up the book and place it in the back compartment of the caddy |
| 6 | LIVING_ROOM_SCENE6 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | LIVING_ROOM_SCENE1 | put both the alphabet soup and the cream cheese box in the basket |
| 8 | KITCHEN_SCENE8 | put both moka pots on the stove |
| 9 | KITCHEN_SCENE6 | put the yellow and white mug in the microwave and close it |

Each task has 50 fixed initial states (0 to 49). Scene `i` loads initial state
`i` with LIBERO env seed `142 + i`, lets the scene settle for 10 no-op steps,
then hands control to the policy. The horizon is 900 steps (45 s at 20 Hz).

## Run one LIBERO-10 rollout

This is the standard Inspect Robots CLI. One rollout of task 2, initial state 0,
with Claude:

```bash
inspect-robots run \
  --task libero_10 -T task_id=2 \
  --embodiment libero \
  --policy agent -P model=anthropic/claude-opus-5-5 -P wire=messages -P effort=low \
  --grader none --no-prompt \
  --log-dir outputs/logs
```

What each part does:

| Flag | Meaning |
|---|---|
| `--task libero_10 -T task_id=2` | LIBERO-10 task 2. Other `-T` options are listed below the table |
| `--embodiment libero` | The LIBERO simulator. It takes the task from the scene |
| `--policy agent -P model=...` | The upstream LLM policy. See [Use a different LLM](#use-a-different-llm) |
| `-P effort=low` | Reasoning effort, forwarded to the provider |
| `--grader none --no-prompt` | Success comes from LIBERO's predicate, so skip the human grader |

`libero_10` task options (`-T`), with defaults:

| Option | Default | Meaning |
|---|---|---|
| `task_id` | `0` | LIBERO-10 task, 0 to 9 (see [LIBERO-10 tasks](#libero-10-tasks)) |
| `start` | `0` | First initial state to run, 0 to 49 |
| `num_scenes` | `1` | Number of consecutive initial states, one episode each. `-T start=10 -T num_scenes=5` runs initial states 10 to 14. `start + num_scenes` must not exceed 50 |
| `max_steps` | `900` | Step limit per episode (45 s at 20 Hz). `-T max_steps=600` gives a 30 s horizon |

Useful `agent` options (all from `inspect-robots-agent`): `-P max_llm_calls=20`
caps LLM requests per episode, `-P image_horizon=2` keeps images only for the
last two observations, and `-P images=on_demand` sends images only when the
model calls `take_pic`.

Useful `libero` options: `-E output_dir=outputs/libero` sets where videos go.
`-E task_notes=true -E task_id=2` adds demonstration-derived hints to the
prompt (see [Few-shot demonstrations](#few-shot-demonstrations)).

Outputs of the run above:

```text
outputs/logs/libero-10-task02_<id>.json          eval log: metrics, actions, LLM usage
outputs/logs/transcripts/, outputs/logs/wire/    agent conversation and raw API traffic
outputs/libero/<time>_task02_init00/rollout.mp4  agentview | wrist video, 20 fps
outputs/libero/worker.log                        simulator stderr
```

Inspect them with the upstream tools:

```bash
inspect-robots inspect outputs/logs/<log>.json --transcript   # what the model saw and said
inspect-robots view outputs/logs --serve                      # HTML viewer
```

A run is a success only if LIBERO's success predicate holds at the end
(`success_at_end = 1`). When the model calls `done`, the episode ends, but
that does not count as success by itself.

## Use a different LLM

Model ids follow the upstream `provider/model` convention. The policy picks
the endpoint from the prefix and the key from the matching variable in `.env`:

| Provider | Example `-P model=` | Key in `.env` | Suggested `-P wire=` |
|---|---|---|---|
| OpenAI | `openai/gpt-6-sol` | `OPENAI_API_KEY` | `responses` |
| Anthropic | `anthropic/claude-opus-5-5` | `ANTHROPIC_API_KEY` | `messages` |
| Google Gemini | `google/gemini-3.7-flash` | `GEMINI_API_KEY` | `chat` (default) |

Example:

```bash
inspect-robots run --task libero_10 -T task_id=0 --embodiment libero \
  --policy agent -P model=openai/gpt-6-sol -P wire=responses -P effort=medium \
  --grader none --no-prompt --log-dir outputs/logs
```

This repo was tested with `openai/gpt-6-astra`, `openai/gpt-6-luna`,
`openai/gpt-6-sol` and `anthropic/claude-opus-5-5`.

## Paid runs with spending caps

`inspect-robots run` has no dollar limit. For experiments that spend real
money, use `scripts/run_budgeted_rollouts.py`. It runs the same task,
embodiment and policy as the CLI, with these additions:

- Budgets are fail-closed. Before each request it reserves a conservative cost
  estimate, and it refuses to send if the episode or total budget would be
  exceeded.
- A request whose cost cannot be confirmed (network error, missing usage)
  blocks all later requests in the run.
- Requests are only sent to the chosen model's own provider endpoint.
- Every request and response body is saved to `api/` for auditing, with a
  per-call cost in `api_usage.jsonl`.
- Each episode gets a fresh policy, so no memory carries over between
  episodes.
- It writes `protocol.json` (everything needed to reproduce the run) and
  `summary.json` (one result row per episode).

```bash
python scripts/run_budgeted_rollouts.py \
  --model anthropic/claude-opus-5-5 \
  --task-id 2 --episodes 5 --start 0 \
  --total-budget 10 --episode-budget 3 --max-calls 20 --effort low \
  --output outputs/task02_opus_zeroshot \
  --execute-paid-rollouts
```

Without `--execute-paid-rollouts` it sends nothing. `--output` must be a new
directory, so a paid run is never resumed and billed twice. Results:

```text
outputs/task02_opus_zeroshot/
  protocol.json   summary.json
  api/api_usage.jsonl, api/ep00_call01_request.json, ...
  episode_00/inspect/<log>.json   episode_00/<time>_task02_init00/rollout.mp4
```

`summary.json` reports two costs, per episode and in total:

| Field | Meaning |
|---|---|
| `estimated_usd`, `total_estimated_usd` | Spend estimated from the reported token usage, with cache reads and writes at their own rates. Close to the provider's bill |
| `budget_charge_usd`, `total_budget_charge_usd` | What the budget caps count. All input is charged at the cache-write rate, so this is higher than the real spend, often several times higher when most input is cached |

Only models with a price row are accepted: `openai/gpt-6-astra`,
`openai/gpt-6-luna`, `openai/gpt-6-sol`, `anthropic/claude-opus-5-5`. To add
one, put its per-million-token prices in `PRICING` in
`src/inspect_robots_libero/budget.py`. A new provider also needs entries in
`PROVIDER_WIRE` and `WIRE_ENDPOINTS`. The provider's own billing is the final
word. The local numbers are conservative estimates.

## Few-shot demonstrations

By default runs are zero-shot. The model gets the instruction, the live
images and robot state, and nothing else. Two optional additions give it
prior knowledge. `protocol.json` records whether either was used
(`"zero_shot": false`).

Demonstration storyboards: 12 frames from each of a few human demos in the
official LIBERO-10 dataset, sent as images before the first observation.
The dataset is about 1.3 GB per task (13 GB for all ten), so download only
the task you need:

```bash
# Official LIBERO-10 demos for task 2 (KITCHEN_SCENE3)
pip install "huggingface_hub[cli]"
huggingface-cli download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset --include "libero_10/KITCHEN_SCENE3_*" --local-dir data/libero

# Storyboards for 2 random demos of task 2
python scripts/prepare_demonstrations.py \
  --task-id 2 --num-demos 2 --seed 42 --dataset-dir data/libero/libero_10
# -> outputs/demos/task02_seed42_n2/manifest.json

python scripts/run_budgeted_rollouts.py --model openai/gpt-6-sol --task-id 2 \
  --demo-manifest outputs/demos/task02_seed42_n2/manifest.json \
  --total-budget 5 --output outputs/task02_sol_2demo --execute-paid-rollouts
```

Task notes: short text hints written by watching one demo. Only tasks 2 and 3
have them, in `src/inspect_robots_libero/notes.py`. Enable them with
`--task-notes` in the paid runner, or `-E task_notes=true -E task_id=N` with the CLI.

## Python API

The same run as the CLI example, written as a script. Run it from the repo root
so `.env` is found.

```python
import os

from inspect_robots import eval
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.defaults import init_dotenv
from inspect_robots_agent.policy import agent_policy
from inspect_robots_libero import libero_10, libero_embodiment

# Load API keys from ./.env into os.environ. The CLI does this for you; eval() does not.
# Variables already set in your shell take precedence over the file.
init_dotenv(os.environ)

task = libero_10(task_id=2, num_scenes=3)                    # initial states 0, 1, 2
embodiment = libero_embodiment(output_dir="outputs/libero")
# No key argument here: agent_policy reads ANTHROPIC_API_KEY from os.environ,
# chosen by the "anthropic/" prefix of the model id.
policy = agent_policy(model="anthropic/claude-opus-5-5", wire="messages", effort="low")
space = embodiment.info.action_space
try:
    (log,) = eval(task, policy, embodiment, log_dir="outputs/logs",
                  # Guardrails: the CLI adds these by default, eval() only if you pass them.
                  approver=ChainApprover(ClampApprover(space), DeltaLimitApprover(space)))
    print(log.results.metrics)                               # {'success_at_end': ...}
finally:
    embodiment.close()                                       # stop the simulator subprocess
```

Two differences from the CLI:

- **API keys.** `eval()` does not read `.env`. Call `init_dotenv(os.environ)`
  at the top of the script, before `agent_policy`, as above.
- **Guardrails.** Without `approver=`, actions go to the arm unchecked. Keep the
  `ChainApprover(ClampApprover(space), DeltaLimitApprover(space))` line.

Like `inspect-robots run`, this has no dollar limit. Use
[`scripts/run_budgeted_rollouts.py`](#paid-runs-with-spending-caps) for paid experiments.

## Implementation notes

- Action space: 10-D absolute end-effector pose in the world frame,
  `x, y, z` (m) plus rot6d (the first two columns of the rotation matrix) plus a
  gripper command in [0, 1] (0 open, 1 closed). The upstream agent's Cartesian
  tools only support rot6d or no rotation, not axis-angle. The worker
  orthonormalises rot6d, converts it to axis-angle, and drives LIBERO's OSC
  controller with `use_delta=False`.
- Observations: `agentview` (flipped on both axes so it is upright, as in the
  standard LIBERO client) and `wrist`, both 256x256. State `eef_state` is the
  controller's own `ee_pos` plus `ee_ori_mat` in rot6d plus a gripper closure
  estimate, so observed and commanded poses share one frame. `robot0_eef_quat`
  uses a different reference frame, so it is not used. `gripper_qpos` is the
  raw finger positions. Object poses and the success predicate are never
  shown to the model.
- Operating notes for the model (frame conventions, gripper behaviour) are in
  `EmbodimentInfo.docs`. The agent policy appends them to its system prompt.
- Success: `success_at_end`, from LIBERO's `check_success()`. The episode
  also ends as soon as the predicate becomes true.

## Repository layout

```text
src/inspect_robots_libero/
  __init__.py      entry-point factories: libero_embodiment, libero_10
  embodiment.py    LiberoEmbodiment (EmbodimentBase), talks to the worker over stdio
  _worker.py       simulator subprocess: LIBERO env, action conversion, video
  tasks.py         libero_10 task, task table, seeds
  budget.py        BudgetTransport: spending caps, pricing, audit logging
  demos.py         demonstration storyboards as a prompt message
  notes.py         optional per-task hints (not zero-shot)
scripts/
  run_budgeted_rollouts.py   paid runs with caps
  prepare_demonstrations.py  storyboards from LIBERO HDF5
  check_adapter.py           simulator-only check
  check_closed_loop.py       closed loop with a mocked LLM
tests/                       unit tests (python -m unittest)
upstream_commit.txt          pinned inspect-robots commit
requirements.lock.txt        pinned third-party deps of the agent stack
```



## Acknowledgements

Built on [Inspect Robots](https://github.com/robocurve/inspect-robots) and
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
([Hugging Face packaging](https://github.com/huggingface/LIBERO)).

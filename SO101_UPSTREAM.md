# Separate upstream SO-101 benchmark

This benchmark executes the genuine upstream
[`SOArmEmbodiment`](https://github.com/robocurve/inspect-robots-so101/blob/9071de0e6fdbda44336b071544e2f8cee6a53a00/src/inspect_robots_so101/embodiment.py).
The existing `--benchmark hardware` path remains available separately.

| Variant | LLM action | IK | Motor-command executor |
|---|---|---|---|
| Existing hardware | `move_to`: Cartesian meters, pitch/roll radians, gripper 0=open/1=closed | Our URDF planner | Our controller + LeRobot |
| Upstream joint | `move_joints`: five motor angles in degrees, gripper native 0..100 | None in the action path | Upstream `SOArmEmbodiment.step()` |
| Cartesian-to-upstream | Same `move_to` contract as existing hardware | Our URDF planner | Upstream `SOArmEmbodiment.step()` |

Upstream supplies a joint-position interface, not Cartesian FK/IK. Its repository
also bundles a LeRobot checkpoint policy, but we use the existing frontier-LLM
`agent_policy`. No learned VLA checkpoint is involved. Choose Cartesian mode to
retain your end-effector experiment. Native joint mode changes the action
problem presented to the LLM; differences in cost/success cannot be attributed
solely to the adapter implementation.

Both new modes retain the isolated LeRobot device I/O, camera crop, calibration,
bounded startup reset, motor-speed/tracking checks and stop-hold behavior. They
inject that I/O through upstream's public `driver_factory` seam. They use the
real upstream class and its joint clamp, pacing, observations and optional
settling, **not its default hardware-driver factory or optional VLA policy**.
Startup reset packets also go through `step()`; upstream `home_pose` is unset
because the existing setup/reset path owns the startup pose. Exit/emergency
holds use device I/O directly, including after camera/upstream failures.

Native motor packets may be subdivided to respect command-speed limits when
encoders lag the previous goal. If a macro retains the existing gripper target,
that target stays constant during the move instead of reopening toward a finger
position blocked by an object. These are benchmark guards outside upstream.

These are upstream command execution with additional benchmark guards. With
settling configured, this benchmark stops at the first failed settle instead
of proceeding after a timeout. Upstream defaults to no settling. The prepared
local config selects a 2-degree arm tolerance; gripper settling is excluded.

## Versions and environments

`upstream_commit.txt` pins Inspect Robots core. `upstream_so101_commit.txt` pins
SO-101 commit `9071de0e6fdbda44336b071544e2f8cee6a53a00`, before the subsequent
Inspect Robots 1.0 requirement. No upstream source is modified. Source hashes
of four upstream control modules are checked before startup and recorded with
the installed version, class and commit in experiment metadata.

The dependency is installed in `llm-libero`. A separate worker environment at
`.venv-so101-upstream` inherits the existing `lerobot312` packages and adds the
pinned core and SO-101 package. The original `lerobot312` environment is unchanged.
To reproduce:

```bash
conda activate llm-libero
python -m pip install --no-deps -r requirements.so101-upstream.txt
python -m pip install --no-deps -e .

/home/koch/miniconda3/envs/lerobot312/bin/python -m venv \
  --system-site-packages .venv-so101-upstream
.venv-so101-upstream/bin/python -m pip install --no-deps \
  -r requirements.so101-upstream.txt
```

`outputs/so101-upstream.json` selects this worker and copies your existing
serial/camera/profile/reset references: `/dev/ttyACM0`, cameras 4/6, follower ID,
calibration file and startup reset. It also sets `upstream_settle_tolerance`,
`upstream_settle_timeout_s` and `upstream_settle_timeout_budget`. Source
repositories and physical calibration files remain read-only.

## One offline trial

```bash
cd /home/koch/LLM_LIBERO
conda activate llm-libero

python scripts/run_so101_upstream.py \
  --catalog outputs/replica-scenes/catalog.json \
  --subset ID --task-id 1 --variation 1 \
  --hardware-config outputs/so101-upstream.json \
  --hardware-mode mock --control joint \
  --model openai/gpt-6.1-sol --effort medium \
  --mock --no-video \
  --output outputs/my-upstream-joint-check
```

For Cartesian control use `--control cartesian` and a new output directory.
These commands run the real policy/tool framework and upstream class with
mock devices and local fake HTTP. They make no real API requests or device
connections and do not measure the model's actual performance.

Completed offline example workbooks:

```text
outputs/so101-upstream-joint-offline/evaluation.xlsx
outputs/so101-upstream-cartesian-offline/evaluation.xlsx
```

## Hardware and paid API modes

Device modes and API flags remain independent, as in [HARDWARE.md](HARDWARE.md):

* `--hardware-mode mock`: no real devices.
* `--hardware-mode dry-run`: real observations, proposed paths, zero motor writes.
* `--hardware-mode execute`: physical commands and the configured startup reset.
* `--mock`: fake API replies, rejected with real motor execution.
* `--execute-paid-rollouts`: real API requests, requiring explicit USD caps.
* Neither API flag: a plan/workbook only, with no devices or requests.

Example paid **read-only** trial with an illustrative $5 cap:

```bash
python scripts/run_so101_upstream.py \
  --catalog outputs/replica-scenes/catalog.json \
  --subset ID --task-id 1 --variation 1 \
  --hardware-config outputs/so101-upstream.json \
  --hardware-mode dry-run --control cartesian \
  --model openai/gpt-6.1-sol --effort medium \
  --max-calls 3 --per-run-budget 5 --total-budget 5 \
  --output outputs/my-upstream-cartesian-readonly \
  --execute-paid-rollouts
```

After checking the real connection/plans, physical execution uses
`--hardware-mode execute`. Cartesian mode requires profile verification flags
or explicit `--assume-calibration`. Native joint mode checks motor calibration,
travel and speeds; Cartesian frame verification is unused. Either mode still
requires motor calibration matching the physical EEPROM. No physical trial was
run by this integration's tests.

The parent terminal handles scene setup; the worker maintains camera preview,
stops and deadlines during inference. Stop commands, Ctrl+C, holds and human
labels follow HARDWARE.md. New run IDs end in `_joint` or `_cartesian`; include
that suffix in stop/score paths. Keep a new output directory for each invocation.

## Reporting and tests

The same XLSX/CSV exporter records calls, steps, per-call costs and totals.
`control_interface` is a summary grouping column. `upstream_steps` counts calls
to the real upstream `step()`; `steps` counts confirmed task motor packets. A
camera failure after a confirmed send therefore does not erase an executed
step. Reset and hold writes remain separate. Success is unscored until a human
labels a completed physical trial.

The framework limiter rebases once per new preflight to the exact policy
observation. Within a path it retains command history. Any subsequent bounds/
delta rewrite aborts instead of sending an unvalidated target. Motor velocity,
travel, tracking and freshness checks remain active below the upstream body.

```bash
python -m unittest discover -s tests -v
python -m pytest vendor/inspect-robots-so101/tests -q
```

Tests exercise the actual class/clamp, no IK in native control, IK followed by
upstream steps in Cartesian control, read-only operation, step accounting on
observation failure, both modes at all three efforts through the unchanged
policy with mocked HTTP, and separate unscored hardware reporting. They never
open real serial/cameras or call a paid API.

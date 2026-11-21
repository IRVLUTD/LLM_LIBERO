# SO-101 hardware adapter

For the separate standard-adapter comparison, see
[SO101_UPSTREAM.md](SO101_UPSTREAM.md).

The hardware backend uses the same **Inspect Robots `agent_policy`** as LIBERO
and Replica simulation. The LLM requests absolute end-effector targets through
`move_to`; a device adapter checks IK and converts the resulting path to calibrated
LeRobot motor targets. There is no separate Gemini/Codex conversation controller
in this backend, and no learned VLA underneath it.

The complete path is:

```text
real top/wrist images + encoder-derived tool pose
  → existing Inspect Robots LLM policy and budget transport
  → whole-path preflight hook
  → bounded Cartesian IK and calibrated motor commands
  → measured robot state and fresh camera observations
```

All new code, logs and configuration live in LLM_LIBERO. The existing
`kochIRVL` files are read-only resources. Neither its evaluation code nor its
calibration/profile/reset files are modified. The adapter's initial validation
was offline, including a mock-device run using the supplied real profile,
calibration and startup reset. Physical connection and motion still need to be
validated on the actual arm.

## Environments and your configuration

Run the evaluation from `llm-libero`. Its subprocess uses your existing
`lerobot312` Python for LeRobot, cameras, SciPy and serial communication. The
worker does not require Inspect Robots or any LLM SDK in that environment.

```bash
cd /home/koch/LLM_LIBERO
conda activate llm-libero
python -m pip install --no-deps -e .
```

Your working prototype command was translated into the local file:

```text
outputs/so101-hardware.json
```

It contains these settings:

```json
{
  "schema_version": 1,
  "python": "/home/koch/miniconda3/envs/lerobot312/bin/python",
  "profile_file": "/home/koch/Desktop/kochIRVL/real_robot/profile.local.json",
  "calibration_file": "/home/koch/Desktop/kochIRVL/calibration/robots/so101_follower/so101_follower_arm.json",
  "port": "/dev/ttyACM0",
  "follower_id": "so101_follower_arm",
  "top_index": 4,
  "wrist_index": 6,
  "width": 640,
  "height": 480,
  "fps": 30,
  "zoom": 1.5,
  "preview": true,
  "video": true,
  "reset_mode": "fixed",
  "reset_action_file": "/home/koch/Desktop/kochIRVL/real_robot/arm_reset.startup.json"
}
```

Relative resource paths resolve beside the configuration file. The profile's
URDF resolves beside the profile, preserving your current setup. Camera
identifiers can be indexes or device paths supported by the installed camera
driver. Change this **local configuration** when USB identifiers change. No
hardware mode is enabled merely by writing or reading the configuration.

Your original execution command used `--assume-calibration`. Although you have
confirmed physical verification, the supplied profile file still records
`kinematics_verified:false` and `world_frame_verified:false`. This adapter
preserves those fields, supports the same explicit assumption flag, and records
both the original flags and the assumption in logs and model notes. Verified
profile files can execute without that flag. Changing a flag itself does not
perform calibration.

## Device mode and API mode are independent

| Option | Meaning |
|---|---|
| `--hardware-mode mock` | Test-double encoders/motors/images; no serial or real cameras. Default. |
| `--hardware-mode dry-run` | Real encoders and cameras; proposed motor paths only. No motor-target or torque writes. |
| `--hardware-mode execute` | Real hardware, bounded motor commands and optional fixed reset. |
| `--mock` | Local fake LLM HTTP responses; zero real API requests. |
| `--execute-paid-rollouts` | Real model requests, with explicit model and USD budgets required. |
| Neither API flag | Write a plan and spreadsheet; no worker, devices, reset, or model requests. |

Fake LLM replies cannot drive real motors: `--mock` combined with hardware mode
`execute` is rejected. Use dry-run to test the LLM/controller pipeline with real
images before enabling execution. Mock-device images are explicitly labeled and
cannot be reported as physical task evaluations.

## Offline check using your existing profile

This exercises the startup reset and a 5 mm requested lift using test doubles;
the named real serial/camera devices in the configuration are ignored:

```bash
python scripts/so101_hardware.py check \
  --config outputs/so101-hardware.json --mode mock \
  --output outputs/checks/so101-offline-01
```

The check has already passed in
`outputs/checks/so101-user-profile-offline/check_report.json`. It confirms FK/IK
and command plumbing with that profile, not its physical calibration accuracy.

A complete offline LLM/adapter trial for your task 1, scene 1:

```bash
python scripts/run_replica_evaluation.py \
  --benchmark hardware \
  --catalog outputs/replica-scenes/catalog.json \
  --subset ID --task-id 1 --variation 1 \
  --hardware-config outputs/so101-hardware.json \
  --hardware-mode mock \
  --model openai/gpt-6.1-sol --effort medium \
  --mock --no-video \
  --output outputs/so101-offline-01
```

The existing offline result workbook is at
`outputs/so101-task01-scene01-offline/evaluation.xlsx`. Its API token/cost values
are synthetic and its devices are mocked. The real source repository was not
imported by the worker.

## Read the actual robot without motor writes

This opens the selected serial bus and cameras, checks the existing EEPROM
calibration/position mode, saves camera images and encoder/FK state, then closes:

```bash
python scripts/so101_hardware.py inspect \
  --config outputs/so101-hardware.json \
  --output outputs/checks/so101-inspect-01
```

Inspect skips the fixed reset. It does not change calibration, enable/disable
torque, or send a motor target. The arm must already be supported or holding its
pose. `pose_check.json` reports encoder-derived tool coordinates and the frame
verification flags; it is not an external measurement of the end effector.

To plan a 5 mm upward motion from the measured pose, still without motor writes:

```bash
python scripts/so101_hardware.py check \
  --config outputs/so101-hardware.json --mode dry-run \
  --output outputs/checks/so101-dry-run-01
```

A dry-run fixed reset is only planned. It does not change the measured starting
pose. If that actual pose is outside the configured model/workspace, preflight
rejects it; use your established setup procedure to put the real arm into its
validated starting pose before checking Cartesian control.

## One paid task 1, scene 1 trial

This example requests GPT-6.1 Sol medium with an example $5 API cap. It requires
`OPENAI_API_KEY` in LLM_LIBERO's `.env` or the environment. Rates are the existing
repository price snapshot; `--pricing-file` can provide confirmed rates.

Start with real images and **no motor writes**:

```bash
python scripts/run_replica_evaluation.py \
  --benchmark hardware \
  --catalog outputs/replica-scenes/catalog.json \
  --subset ID --task-id 1 --variation 1 \
  --hardware-config outputs/so101-hardware.json \
  --hardware-mode dry-run \
  --model openai/gpt-6.1-sol --effort medium \
  --max-calls 3 --max-steps 1800 --policy-seconds 900 \
  --per-run-budget 5 --total-budget 5 \
  --output outputs/so101-task01-scene01-dryrun-01 \
  --execute-paid-rollouts
```

For actual motor execution after validating the connection/plans, use the same
command with:

```text
--hardware-mode execute --assume-calibration
--max-calls 80
--output outputs/so101-task01-scene01-execute-01
```

Keep the remaining arguments, including `--execute-paid-rollouts` and budgets.
This opts into the fixed startup reset specified by your configuration. The
startup reset is interpolated from measured motor positions and checked against
calibrated travel/speed limits. It is logged separately from task motor steps.

The preview shows real top/wrist images, with the task reference overlaid on the
top view for the operator. The overlay is **never sent to the model**. After
reset/setup, arrange the objects and press Enter to start. `--auto-start`
explicitly skips that wait. Without it a real-device trial requires an
interactive terminal. Set `preview:false` in a local config for operation without
a desktop display; this does not skip operator scene setup.

For task 2 scene 1, change `--task-id 1` to `--task-id 2`; `--variation 1` remains
the first reference setup. Repeated model/effort selectors and scene selectors
work as in simulation. Each hardware episode requires physical scene setup;
the JSON scene's simulated object poses and MuJoCo success predicates are unused.

## What the adapter checks

The existing policy's `pre_check` hook receives the entire interpolated Cartesian
tool path. The worker rejects out-of-workspace targets, overlong translations,
expired observations, drift, unreachable IK, calibrated motor travel violations,
and paths that exceed the remaining motor-step budget. Motor interpolation is
also checked in Cartesian space and subdivided to obey the profile's motor
velocity limits. No feasible prefix of a rejected path is executed.

Only waypoints matching the approved path and its current plan ID may execute.
The framework limiter uses the policy's source observation as its reference
once per new accepted path, then limits each waypoint from the previous command
within that path. This avoids rewriting a new path against an older requested
pose when physical tracking differs. A post-preflight modification or protocol
mismatch stops the episode instead of repeatedly returning no motion.
The first command rechecks observation age and drift after preflight. Every
motor waypoint checks measured tracking; clamped commands retry the same
waypoint, and catch-up has a bounded timeout. A grasp retains its commanded
closure even if the fingers stop apart on an object. Real cameras are marked
uncalibrated until actual transforms are supplied; simulated camera poses and
world-axis arrows are not used on real images.

The profile defines the motor degree/radian mapping, fingertip offset, world
frame, workspace, speeds, tracking tolerance, and maximum observation age. The
supplied profile uses 20 Hz, 2 cm/s Cartesian speed, 15 deg/s arm speed, 5 cm per
move, and a 60 s observation-age limit. These are software motion bounds, not
force control or collision avoidance. A numerical IK preflight does not model
contacts with real objects or people.

Use the shared evaluation runner: it wires the policy's preflight hook. The
registered `so101_hardware` embodiment blocks ordinary motion when no approved
path exists; it cannot be driven by a generic CLI policy without that hook.

## Stop and shutdown

Ctrl+C in the evaluation terminal requests shutdown. Pressing `q` or closing
the camera preview requests a stop. From another terminal, use the exact
episode hardware directory:

```bash
python scripts/so101_hardware.py stop \
  --episode outputs/so101-task01-scene01-execute-01/ID_task01_v01_r01_m00_medium/hardware
```

The worker checks its `STOP` file and wall-time deadline during motion and while
the LLM is waiting. A stop requests a measured arm-position hold while retaining
the last gripper target, and ends the hardware sweep. It does not release an
object or automatically move to reset. API inference may still finish and be
billed after motor execution has stopped. Independent hardware workers lock the
serial port against other workers from this integration; close other robot
control programs before using it.

Worker shutdown preserves torque/hold. For `run_replica_evaluation.py` physical
execute episodes (both hardware benchmarks), the runner now saves results and
then prompts:

```text
[END] Episode finished. Support the arm and remove/support any held object before releasing torque.
Press ENTER to disable torque on all six SO-101 motors, or type keep to retain the hold:
```

This applies after model `done`, `give_up`, budget/step stops and evaluation
errors whenever a hardware worker was opened. Enter reconnects to the same
serial port solely to disable torque and verify `Torque_Enable=0` on all six
motors, including the gripper. It never enables torque, sends a position goal,
opens cameras, or changes calibration/PID settings. Calibration must still
match the selected arm. The outcome is saved in `hardware/torque_release.json`,
`hardware_summary.json`, and the evaluation spreadsheet's
`torque_release_status` / `torque_disabled` fields. Release failure is reported
and stops a multi-episode sweep.

Typing `keep`, EOF, or a noninteractive terminal preserves the existing hold.
`--keep-torque` skips this end prompt explicitly; `--auto-start` only skips the
startup setup prompt. If another episode remains after confirmed release, the
runner waits for Enter before reconnecting and re-enabling torque for that
episode. End-prompt waiting and release time are excluded from the episode's
reported wall time. Mock/dry-run episodes do not prompt or release torque.
The separate trajectory replay script retains its existing hold-on-exit behavior.

Software stop handling does not replace a physical emergency stop, and a lost
serial connection cannot guarantee delivery of a hold or torque-release command.

## Multi-scene agent session

`scripts/run_vla_replica_agent.py` is a separate terminal-operated program for
the following ordered ID queue, configured in
`configs/vla_replica_agent_session.json`:

| Task | Scenes |
|---|---|
| 2: bowl on coaster | 1, 3, 4 |
| 4: fold towel | 1, 2, 5 |
| 6: erase whiteboard | 1, 2, 4 |
| 9: press button | 1, 3, 4 |
| 10: blocks in box | 1, 3, 4 |

Task instructions, counts, colors and references come from the selected catalog
scenes. Each attempt uses the existing hardware benchmark and a new LLM
conversation. Hindsight from previous tasks is not supplied to later tasks.

Create a plan without devices or API calls:

```bash
python scripts/run_vla_replica_agent.py \
  --hardware-config outputs/so101-hardware.json \
  --model openai/gpt-6.1-sol --effort medium
```

An actual session uses explicit hardware/API modes and budgets. These $5/$75
caps are examples, covering all retries within the session-wide limit:

```bash
python scripts/run_vla_replica_agent.py \
  --hardware-config outputs/so101-hardware.json \
  --hardware-mode execute --assume-calibration \
  --model openai/gpt-6.1-sol --effort medium \
  --max-calls 80 --max-steps 1800 --policy-seconds 900 \
  --per-run-budget 5 --total-budget 75 \
  --execute-paid-rollouts
```

Select `outputs/so101-hardware-fast.json` instead if you want the separately
configured faster motion limits. `--initial-pose file.json` overrides the
hardware config's `reset_action_file`; otherwise the existing startup pose is
used. Initial pose values are validated against calibrated motor travel. The
session never performs an automatic startup reset: each reset needs `r+Enter`.

Terminal workflow:

1. The reference overlay and live cameras open during **SETUP**. Clear the reset
   path and remove/support held objects, then press `r+Enter` to reset. Arrange
   the scene against its reference and press Enter to start the agent.
2. During the attempt the operator preview shows clean camera images. The LLM
   receives live images without the reference overlay. Pressing `q` in the
   preview stops the worker; Ctrl+C stops the session.
3. After model termination, the reference overlay returns. Enter `s` for human
   confirmed success, `f` for failure, or `u` to leave it unscored. Model `done`
   alone never becomes a physical success label.
4. After success, `r` resets the arm and `n` or Enter advances to the next scene.
   After failure/unscored results, `r` resets, `t` retries the same scene with a
   fresh conversation, and `k` skips ahead. Reset is required before starting
   the next attempt. Each retry gets a new folder and consumes the same session
   budget. At setup, `k` skips an unstarted scene without recording a failure.
5. Hardware faults pause for operator action. There is no automatic retry or
   new task motion after a fault. Unknown API usage blocks subsequent attempts.
   `q` ends the session. A connection failure offers explicit reconnect or quit.
6. Torque stays enabled between scenes. At session end the existing Enter-to-
   disable-torque prompt runs once. `--keep-torque-at-end` retains the hold.

Human setup/scoring/reset waits do not consume the agent's episode deadline.
Setup/reset videos, trajectories and motor counters live separately under
`operator/`; they are not added to the agent's action-step counts. The default
video recording includes both operator stages and individual attempts.

Every invocation creates a new timestamped folder such as:

```text
outputs/VLAReplica_Agent/20261120_173747-0600_openai_gpt-6.1-sol_medium/
  evaluation.json
  evaluation.xlsx
  episodes.csv
  calls.csv
  by_reasoning.csv
  session_events.jsonl
  human_labels.jsonl
  hardware_profile.json
  initial_pose.json
  scenes/ID_task02_v01/scene.json
  scenes/ID_task02_v01/reference.jpg
  operator/001_ID_task02_v01_SETUP/
  task02_scene01/attempt01/
    evaluation.xlsx
    ID_task02_v01_r01_m00_medium/
      trajectory.json
      api/
      inspect/
      hardware/rollout.mp4
      hardware/motor_events.jsonl
  task02_scene01/attempt02/
  task02_scene03/attempt01/
```

Folder timestamps default to `America/Chicago` (`--timezone` can change this),
and `--agent` optionally supplies a shorter agent label. Photos, profile, URDF,
calibration and reset pose are copied into the session folder for reproducible
configuration. The source kochIRVL repository is read-only. Session-wide reports
retain every attempt, its calls/token costs, action counts and human label;
per-attempt reports remain available as well. `--queue` selects a different
explicit queue if needed. Interrupted sessions retain completed/partial results
and mark unstarted attempts as not run.

For offline checks use `--mock --hardware-mode mock`. `--operator-script` accepts
a JSON array of terminal answers only with both mock devices and mock API
responses. Mock arm state persists between its setup and episode workers, but
it is not a physical simulation. Human mock/dry-run verdicts are logged as
operator checks and never counted as physical success. No live hardware or
paid API request is needed to test the workflow.

## Investigating a shoulder tracking stop

If a lift works during teleoperation but stops in the benchmark with
`tracking_failed`, inspect the last `waypoint_result` in `motor_events.jsonl`.
It names the motor, measured position, last command and rejected next waypoint.
The two-second catch-up timeout is bounded waiting at the existing goal; it
does not increase actuator effort. Repeated stops need a comparison of command
timing and servo configuration, rather than simply extending that timeout.

The `inspect` command also writes `motor_diagnostics.json`:

```bash
python scripts/so101_hardware.py inspect \
  --config outputs/so101-hardware.json \
  --output outputs/so101-motor-diagnostics-01
```

Close other serial-control programs first. Inspection reads PID coefficients,
torque limits, goal/actual position, velocity, current, voltage, temperature and
status without writing goals, gains, torque or calibration. Values are register
values with sign decoding and no calibration normalization; do not interpret
raw current/load values as calibrated forces. Each failed register read is
recorded under `read_errors`. This snapshot is taken during inspection, not
during the previous failed motion, and cannot retrospectively identify a
transient fault.

This adapter preserves the servo configuration already on the arm. Its
connection routine deliberately skips LeRobot's `configure()` method. Compare
the readout against a session prepared using your successful teleoperation
procedure before changing gains or torque limits.

For the reported task-1 lift, local copies were created at
`outputs/so101-hardware-slow.json` and
`outputs/so101-hardware-slow.profile.json`. They halve Cartesian translation,
angular and arm-joint speed to 0.01 m/s, 0.2 rad/s and 7.5 deg/s, respectively.
Both `outputs/so101-hardware.json` and the slower configuration now use local
profile copies with a 12-degree tracking limit, as requested. This allowance
applies to all five arm joints. Gripper settings and calibration flags are
unchanged, and the kochIRVL repository is untouched.
These are a diagnostic trial configuration, not a confirmed hardware fix.
Use your evaluation command with these replacements:

```text
--hardware-config outputs/so101-hardware-slow.json
--max-steps 3600
--output outputs/so101-task01-scene01-slow-01
```

The larger step allowance accommodates the more finely sampled slow path;
the 900-second wall-time limit and $5 API limits remain unchanged. If the joint
still fails to converge at the held goal, slowing the path is insufficient.
Compare servo settings and measured responses with the successful teleoperation
baseline before tuning motor gains. A larger tracking allowance by itself does
not restore accurate end-effector positioning.

## Replay simulation actions without an LLM

`scripts/replay_replica_trajectory.py` reads a single saved simulation episode.
`--track commands` (default) replays the Cartesian action requests saved by the
framework; those can include targets rejected by the simulation IK.
`--track observed` instead follows achieved poses from `trajectory.json`.
Neither mode copies simulator joint radians directly into hardware motor degrees.
The path is solved again with hardware IK and retimed to hardware speed limits.
Inference pauses are omitted, and no LLM calls or API charges occur.

Offline preparation opens no cameras or serial devices:

```bash
python scripts/replay_replica_trajectory.py \
  --source outputs/task02_scene01_sol61_medium_01 \
  --hardware-config outputs/so101-hardware-slow.json \
  --max-source-steps 22 \
  --output outputs/task02-scene01-replay-prepare-01
```

With the current profile and an unverified identity frame mapping, the full
721-step source run fails preflight: 61 requested poses are below the hardware
20 mm floor, with a minimum of 8.33 mm. Checking the above-floor prefix also
rejects source step 23 at 0.1014 rad orientation error against the 0.1 rad limit.
The first 22 source requests pass offline planning from the configured startup
pose: 468 motor commands (including approach), at least 23.4 seconds. This short
prefix covers initial approach/lifting/opening motions, not a full placement.
The source simulation's scorer reported failure, despite the model calling done.

After arranging the bowl and coaster, the selected physical prefix test is:

```bash
python scripts/replay_replica_trajectory.py \
  --source outputs/task02_scene01_sol61_medium_01 \
  --hardware-config outputs/so101-hardware-slow.json \
  --mode execute --assume-calibration \
  --max-source-steps 22 --time-scale 2 \
  --output outputs/task02-scene01-hardware-replay22-01
```

The runner disables the fixed startup reset in a derived local configuration.
Execute-mode connection establishes the existing measured-position hold; all
replay/approach motion waits for a fresh full-path preflight and interactive
`REPLAY` confirmation. The live plan starts from measured robot state, so its
length or feasibility may differ from offline planning. An expired start
observation or subsequent robot drift invalidates the plan. Other controllers
must be closed. Camera preview, video, STOP file, tracking limits and wall-time
checks use the existing hardware pipeline. There is no collision planner.

`--mode dry-run` opens devices read-only and validates against measured state;
`--mode mock` exercises the bridge on fake devices. The default `prepare` mode
stays entirely offline. Use a new output directory for every invocation.

Resume a recording with `--start-source-step 23`. This is one-based and retains
the original source step numbers in plans, rejections and tracking CSVs.
`--max-source-steps` limits the number of samples starting at that selected step.
Offline validation can start from a saved motor pose via `--start-motors`;
physical execution always replans from fresh measured state.

When explicitly requested, `--retarget pitch-height` generates an adjusted path
before the strict whole-path check. It preserves X/Y, tool roll and gripper
commands, raises low heights above the existing floor with a position-error
margin, and selects feasible tool pitch through hardware IK with a one-degree
joint margin. It does not relax the runtime IK tolerance, calibrated motor
travel, speeds, tracking limit or height floor. `retarget_report.json` lists all
original/adjusted poses and maximum changes, and `retargeted_targets.json` saves
the adjusted Cartesian path. The original recording is untouched.

For the explicitly authorized continuation of task 2, the replay-only profile
`outputs/so101-hardware-replay-rest.profile.json` extends the modeled wrist upper
bound to its calibrated +102.54945 degrees. This is based on the user's assumed
clearance, not a new physical calibration. The remaining 699 source samples
require pitch changes up to 13.6 degrees and height increases up to 17.7 mm;
the lowest adjusted target is 26 mm. The original 20 mm workspace floor and
12-degree tracking guard remain. Its command is:

```bash
python scripts/replay_replica_trajectory.py \
  --source outputs/task02_scene01_sol61_medium_01 \
  --hardware-config outputs/so101-hardware-replay-rest.json \
  --mode execute --assume-calibration \
  --start-source-step 23 \
  --start-motors outputs/so101-replay22-last-measured.json \
  --retarget pitch-height --time-scale 2 \
  --output outputs/task02-scene01-hardware-replay-rest-adjusted-02
```

Frame mapping is explicit and editable through `--frame-map path/to/map.json`:

```json
{
  "schema_version": 1,
  "verified": false,
  "translation_m": [0, 0, 0],
  "yaw_deg": 0,
  "description": "Replace with measured simulation-world to hardware-world alignment"
}
```

Default identity mapping is labeled unverified; execute mode requires a verified
mapping or explicit `--assume-calibration`. Translation/yaw transform positions
between Z-up frames; geometric pitch/roll and normalized gripper closure retain
their meanings. No height offsets, workspace changes or relaxed IK thresholds
are introduced automatically. `mapped_object_origins` in `replay_manifest.json`
lists the scene object origins in the selected hardware world frame. These are
provisional simulation positions, not measurements of the physical scene.

Each preparation saves source/profile/calibration/code hashes, `frame_map.json`,
`source_targets.csv`, a full `offline_preflight.json`, and (when valid)
`offline_plan.json` plus `motor_targets.csv`. Physical/live-state planning also
saves `hardware/replay_plan.json` and `hardware/replay_preflight.json`.
`hardware/replay_tracking.csv` compares each sent command with encoders read
after that control tick, including arm angular errors and encoder-FK Cartesian
errors. These are tracking measurements, not externally measured tip accuracy.
`hardware/replay_summary.json` records completion/stops, peak arm error, zero
LLM calls/cost, and unknown physical task success. Standard motor events,
hardware snapshots, STOP handling and video are retained.

## Artifacts, step definitions, and success labels

The existing `evaluation.xlsx` and CSV/JSON reports include API calls, retries,
per-call token costs, total cost, reasoning effort and timing. Added fields:

| Field | Meaning |
|---|---|
| `device_mode` | Mock devices, read-only dry-run, or physical execution |
| `steps` | Actually sent task motor waypoints, including command retries; zero for dry-run |
| `planned_control_steps` | Motor waypoints corresponding to consumed approved Cartesian waypoints |
| `framework_steps` | Framework action invocations, including model termination |
| `reset_control_steps` | Startup reset motor waypoints, separate from task steps |
| `motor_writes` | All confirmed driver target calls, including setup, task and holds |
| `hold_writes` | Hold target calls, including the initial connection hold |
| `commanded_motion_seconds` | Task motor steps / control frequency; excludes inference/catch-up waiting |
| `human_success` | Explicit human verdict, initially blank |
| `environment_success` | Blank for hardware; no simulator oracle |

Hardware has no `simulated_seconds`. Grouped results separate device mode as
well as benchmark/model/effort/calibration/accounting mode. Unscored physical
episodes do not appear as measured failures or zero successes.

Each episode saves `hardware_snapshot.json` (profile/calibration/URDF/reset
hashes and settings), `motor_events.jsonl` (planned/requested/sent/measured motor
data), preflight results, pose images, the framework transcript/actions, a real
wall-time video at approximately 10 fps, and `hardware_summary.json`. API keys
are kept in the parent process and are not passed to the hardware worker.

After inspecting a finished physical trial, label it explicitly:

```bash
python scripts/so101_hardware.py score \
  --evaluation outputs/so101-task01-scene01-execute-01 \
  --run-id ID_task01_v01_r01_m00_medium \
  --success yes --note 'Bread placed and released on the red plate.'
```

Use `no` or `unknown` as appropriate. The label updates the workbook and keeps an
append-only annotation history. It preserves the original model outcome;
`model_done_unverified` is never converted into simulator-confirmed success.
Mock/dry-run episodes cannot be labeled as physical success/failure.

## Offline verification

```bash
python -m unittest discover -s tests -v
```

Hardware tests cover full-path rejection before motion, motor speed/travel
bounds, observation freshness and drift, post-preflight drift, read-only plans,
grasp-command retention, the unverified-profile execution gate, stop handling
during model idle time, all three effort settings through the real upstream
policy using local fake HTTP, and separate human scoring. No test opens a real
serial device or camera or sends a live model request.

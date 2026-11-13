# VLA Replica SO-101 simulation

This integration runs VLA Replica scenes and LIBERO with the same Inspect Robots
agent policy, reasoning-effort parameter, budget transport, and XLSX/CSV exporter.
All preparation, simulation, and results live in **LLM_LIBERO**. The external
`kochIRVL` repository supplies read-only task instructions and reference images;
it is neither imported nor modified. No robot hardware modules are loaded.

The starting scenes are **editable approximations**, not faithful reconstructions.
There are 50 ID, 40 OOD, and 15 additional AB variants in the local task folder.
Layouts, distractor choices, geometry, and camera poses require calibration before
claiming equivalence to real-world benchmark results. The paper defines the 90
ID/OOD scenes; AB comes from the local repository. Low/medium/high here refer to
model reasoning effort, independently of task difficulty and ID/OOD/AB subset.

## Install and prepare

From the LLM_LIBERO root, use its existing Python environment:

```bash
conda activate llm-libero
python -m pip install -e '.[replica]'
python scripts/prepare_replica_scenes.py \
  --tasks-root /home/koch/Desktop/kochIRVL/tasks \
  --output outputs/replica-scenes
```

Preparation requires MuJoCo to validate and build keyframes. It does not call an
LLM. `--output` must be new so existing edits cannot be overwritten. The prepared
catalog and scenes already exist at `outputs/replica-scenes` in this workspace.

Each scene folder contains:

```text
ID_task02_v03/
  scene.json   editable objects, cameras, robot mounting, scoring, provenance
  scene.xml    runnable MuJoCo model
catalog.json   (in the parent directory) ordered scene entries
```

JSON variation 1 corresponds to `pic1`, variation 2 to `pic2`, and so on, even
when instruction strings repeat. The importer accepts a list of five strings or
`{"tasks": [...]}` containing strings or objects with `name`/`task` fields. It
records hashes and paths for both the instruction file and reference photograph.
Photos are setup references; they are not policy demonstrations or calibration.

## Inspect a scene without an API call

```bash
python scripts/check_replica.py \
  --catalog outputs/replica-scenes/catalog.json \
  --scene-id ID_task02_v03 --viewer \
  --output outputs/checks/replica-view
```

The check saves the initial top/wrist images, holds the pose, and raises the tool
1.5 cm through IK and physics. The separate viewer opens the compiled scene and
its controls; close it to return. Without `--viewer`, the check is headless.
To check every generated scene, omit `--scene-id` and use `--all-scenes`. By
default it checks one scene per task family. Every check needs a new output path.
This tests infrastructure and initially unsolved scenes, not agent task completion.

The evaluation runner also accepts `--viewer` for Replica. It opens a live
desktop window that mirrors the actual evaluation state and paces motion at
20 Hz. The window stays responsive while the model is reasoning; simulation
time pauses during inference. Closing it ends the episode before the next
physical action (an in-flight API request may still finish). Omitting `--viewer`
keeps evaluation headless. This requires a working desktop `DISPLAY`; viewer
controls change only the displayed copy. Videos and spreadsheets are still saved.

## Run the complete loop offline

```bash
python scripts/run_replica_evaluation.py \
  --catalog outputs/replica-scenes/catalog.json \
  --subset ID --task-id 2 --variation 3 \
  --model openai/gpt-6-luna --mock --no-video \
  --output outputs/my-replica-offline-check
```

This uses real cameras, physics, the upstream agent policy, function tools, and
budget accounting. Only the provider response is mocked: it opens the gripper
then gives up. There are **zero real API calls and no API charges**. Token usage
and costs are synthetic and appear only under the mock/simulated columns. It
does not evaluate that model's actual reasoning or demonstrate task success.

Omit `--mock` and `--execute-paid-rollouts` to generate an **unexecuted plan**.
No model choice is required for a plan; the model column then says
`awaiting-model-selection`. With the full catalog this gives 315 scheduled
episodes (105 scenes × three efforts), with measurement cells left blank.

## Paid evaluation, after choosing models and a budget

The implementation does not start paid runs automatically. An explicit example:

```bash
python scripts/run_replica_evaluation.py \
  --catalog outputs/replica-scenes/catalog.json \
  --subset ID --task-id 2 --variation 3 \
  --model openai/gpt-6-luna \
  --max-calls 80 --max-steps 1800 \
  --max-output-tokens 8192 \
  --per-run-budget 1 --total-budget 3 \
  --output outputs/my-paid-replica-comparison \
  --execute-paid-rollouts
```

Model names above are existing repository options, not a fresh availability or
pricing check. Confirm the selected model and rates before a paid experiment.
The runner inherits the existing `budget.py` price snapshot and records that
fact. A `--pricing-file rates.json` can supply verified per-million-token rates:

```json
{
  "provider/model": [2.0, 0.2, 2.5, 10.0]
}
```

Those illustrative numbers are ordered as regular input, cached input,
cache-write/reservation input, output. The reservation rate must cover both
other input rates. Supported budgeted provider wires are OpenAI Responses and
Anthropic Messages. Reusing a different provider needs corresponding transport
and accounting support; Gemini's real-world prototype is not imported here.
No credentials are copied from `kochIRVL`; paid keys come from LLM_LIBERO's `.env`
or the environment. Reports never persist headers or API keys.

Repeat `--model` for multiple models. Low, medium, and high are the default;
repeat `--effort` to choose a subset. Select multiple scenes with repeated
`--subset`, `--task-id`, `--variation`, or `--scene-json` flags. `--repeat N`
repeats each fixed setup. Each episode has a fresh policy and simulator. All
models/efforts use the same initial scene; hashes allow checking that comparison.

Budgets cover the entire invocation and each episode. Requests reserve their
full output-token cap plus conservative input usage before transmission. Unknown
usage stops the entire sweep; its reservation stays charged. A provider rejection
is recorded, rather than silently changing the model or effort. The output
directory must be new; the runner never resumes and rebills an existing run.

For a directly comparable LIBERO experiment, use the **same runner**:

```bash
python scripts/run_replica_evaluation.py \
  --benchmark libero --task-id 2 --init-state 0 \
  --model openai/gpt-6-luna --mock \
  --max-steps 1800 --output outputs/my-libero-offline-check
```

LIBERO IDs retain their native 0..9 numbering. Replica IDs use the external
repository's 1..10 filenames. Both adapters control at 20 Hz. Their robot
kinematics, task predicates, and action dimensions differ; shared policy,
accounting, and matched explicit budgets do not make their tasks equally hard.
Older LIBERO commands in the main README remain available.

## Spreadsheet and raw data

Each invocation writes `evaluation.xlsx`, `episodes.csv`, `calls.csv`,
`by_reasoning.csv`, `evaluation.json`, and `report_data.json`. The workbook has
Episodes, Calls, By reasoning, Low, Medium, High, and Protocol sheets, with frozen
headers and filters. It requires no spreadsheet Python dependency.

The episode rows identify task, instruction, subset, variant, repeat, model,
reasoning effort, fidelity, and calibration status, alongside:

| Field | Meaning |
|---|---|
| `llm_calls` | Logical model requests, including requests with unconfirmed usage |
| `http_attempts` | Actual wire attempts, including transport retries |
| `real_api_calls` / `mock_calls` | Distinguishes live requests from offline tests |
| `steps` / `simulated_seconds` | 20 Hz control ticks and ticks / 20; initial settling excluded |
| `estimated_cost_usd` | Complete episode's usage-based API cost, blank if any usage is unknown |
| `known_cost_subtotal_usd` | Known costs only; a lower bound when usage is incomplete |
| `mean_cost_per_call_usd` | Complete total divided by logical calls |
| `reasoning_output_cost_usd` | Optional reasoning-token subset of output cost; already included in total |
| `simulated_cost_usd` | Mock accounting only; never actual API spending |
| `budget_charge_usd` | Conservative cap accounting, including unknown reservations |
| `environment_success` | Private physical/proxy predicate, independent of the model saying `done` |

The Calls sheet includes the individual call cost, token counts, optional
reasoning tokens, latency, and HTTP status. The reasoning summary separates
benchmark, model, effort, fidelity, calibration, and accounting mode. Empty cells
mean unavailable or unrun measurements. They do not imply free requests. Agent
cost is the sum of all API token costs, including reasoning; local GPU/electricity
and subscription usage are excluded. Provider billing remains authoritative.

Each episode also saves the exact JSON/MJCF scene, source/mesh hashes, initial
images, a robot trajectory, private scorer diagnostics, framework observations,
actions and transcripts, API audit bodies/usage events, and a 20 fps rollout
video. `--no-video` disables Replica videos. The protocol records the repository
commit, pinned upstream commit, exact embodiment notes and their hash, rates,
settings, and scene snapshots. To rebuild spreadsheets without network calls:

```bash
python scripts/run_replica_evaluation.py \
  --output outputs/my-replica-offline-check --report-only
```

## Edit geometry and calibrate layouts

Edit a scene's `scene.json`, then explicitly recompile it:

```bash
python scripts/prepare_replica_scenes.py \
  --compile outputs/replica-scenes/ID_task02_v03/scene.json
```

Compilation replaces that scene's XML only. Evaluation reads the existing XML
without regenerating it, so direct MJCF edits also survive runs. Keep object and
joint names consistent with the scorer. Change `scoring` values to update its
body references and thresholds. Body names specific to articulated task types
(`towel_segmentN`, `oven_door`, `button_cap`, `ink_i_j`) are part of their scorer
contract and must be preserved or their scorer implementation updated.

Object fields include `pos` in world meters, optional `quat` (w,x,y,z) or `euler`
(radians), `size`, `mass`, `fixed`, `color`/`rgba`, and `kind`. Box `size` values
are half extents; cylinder sizes are radius and half height. Bowl sizes are
radius and total wall height. Edit `robot.base_pos`, `robot.base_euler`, and
`robot.home`, and add entries in `cameras` for `top_camera` or `wrist_camera` to
change camera attributes. Wrist-camera positions remain relative to the wrist
body, as defined in MJCF. The compass is recalculated from the current camera.

For replacement meshes, add a top-level list:

```json
"meshes": [{"name": "bread_scan", "file": "meshes/bread.obj", "scale": [1, 1, 1]}]
```

Set the relevant object's `kind` to `mesh` and `mesh` to `bread_scan`, retaining
its `name`, `pos`, `size`, and color fields. Relative mesh files resolve beside
`scene.json`. For multipart collision shapes or joints, set an object's
`custom_mjcf` to a file containing a single `<body name="...">...</body>` with
the same object name, or edit the generated XML directly. Add required mesh
assets to the top-level `meshes` list or XML `<asset>` section. Mesh files should
remain available; the run records mesh hashes but does not bundle custom files.

Use the recorded reference image for overlay/calibration work. When geometry,
camera extrinsics, and object placements have actually been validated, update
`calibration.status`, its `notes`, and `fidelity` to describe the evidence. Merely
editing those labels does not perform calibration. Keep original source files
untouched and create local scene copies when comparing calibrated revisions.

## Scoring approximations and task IDs

| Repository ID | Task | Scoring implementation |
|---|---|---|
| 1 | Bread on plate | Contact-supported, released, still, centered |
| 2 | Bowl on coaster | Contact-supported, upright, released, still, centered |
| 3 | Stack blocks | Support contact, upright, released, still, centered |
| 4 | Fold towel | Articulated strip endpoint proximity after robot contact; **cloth proxy** |
| 5 | Open oven | Hinged door threshold after robot contact |
| 6 | Clean whiteboard | Eraser contact clears surface patches; **coverage proxy** |
| 7 | Pepper N shakes | Grasped inversion/upright cycles above target plate; **no-grain proxy** |
| 8 | Lift bowl N times | Robot-contact lift and return cycles, with height hysteresis |
| 9 | Press button N times | Contact-driven spring-loaded depression/release cycles |
| 10 | Collect N blocks | All specified blocks supported/released/still inside the box |

All success predicates require a configurable dwell (default 0.5 s). Counted
tasks require exact counts and complete cycles; holding a button down or a bowl
aloft is not counted repeatedly. Overshooting loses success. These are new
simulation predicates, not validated equivalents of the paper's real-world
judgments. Task numbering in paper Table 3 differs; the scene JSON records
`paper_task_number` separately (the appendix and repository use another order).
An LLM's `done` call alone cannot establish success.

The SO-101 adapter uses six labeled Cartesian dimensions: x, y, z, pitch, roll,
gripper. Pitch/roll follow the existing prototype's five-joint arm convention,
not independent 6-DoF rotations. Upstream interpolation uses `rotation_repr=none`
and these labels; the embodiment notes define their absolute geometric meaning.
IK and motor commands move only through MuJoCo physics. Rejected waypoints spend
a control tick, so repeated invalid moves cannot bypass the horizon. The model
sees RGB and robot proprioception, including numeric controller feedback. Task
object coordinates, counts and scorer diagnostics remain private.

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/check_replica.py --catalog outputs/replica-scenes/catalog.json \
  --all-scenes --output outputs/checks/replica-all
```

New tests cover read-only source handling, ordered variants, geometry edits,
physical movement, supported placement, exact cycle counts, the real upstream
policy using mocked HTTP at all three efforts, XLSX XML integrity, and sweep
termination on unknown usage. They never require a live key.

Source context: `/home/koch/Downloads/VLAReplica_NeurIPS_Final.pdf`, especially
sections 3.3–3.4 and Appendix C. Arm geometry and the IK/rendering foundation
were copied from the user's existing `kochIRVL/sim` prototype; see
`src/inspect_robots_libero/replica/assets/NOTICE.md` and its Apache-2.0 asset license.

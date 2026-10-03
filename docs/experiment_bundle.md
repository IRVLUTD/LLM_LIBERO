# Upload experiment numbers to Google Drive

The script exports **one JSON file** with the current viewer's results across every model, reasoning effort, task, and initial state. It reads the campaign manifest, summaries, manual review outcomes, and small API usage logs directly from metadisk. It includes episode outcomes, steps, costs, simulator metrics, call counts, token counts, call latency, HTTP status, and numerical summaries by model/effort/task. Superseded runs are excluded as in the viewer. Missing episodes and unknown outcomes are counted separately; unavailable token usage is null at episode/call level. Aggregate sums include only known measurements. Success rate uses episodes with known official outcomes.

```bash
# Export locally:
python scripts/upload_experiment_bundle.py --export-only
# Export and upload:
python scripts/upload_experiment_bundle.py
```

Default source: `/metadisk/may/llm_libero_experiments/libero_experiments_20260930`.
Default output: `outputs/experiment_results.json`.
No videos, images, transcripts, prompts, raw API responses, viewer HTML, or archive are included.

Uses the same Drive folder and OAuth token as the video uploader. Install `google-api-python-client` and `google-auth-oauthlib` if needed. On first login add `--credentials outputs/drive_auth/credentials.json`. Options: `--source`, `--output`, `--folder-id`, `--token`, `--dry-run`, and `--export-only`.

Rerunning updates the same campaign's results file in Drive, preserving its file ID. The script prints the Drive link and saves a receipt in `outputs/experiment_results.drive.json`. Sharing follows the destination folder.

In another repo, download the JSON using the printed Drive link and load `episodes` to display individual results, `by_model_task` for comparisons, and `totals` for headline counts. Automated downloads use the printed file ID with `https://www.googleapis.com/drive/v3/files/FILE_ID?alt=media` and an OAuth bearer token with file access.

## Response points and motion visualization

Export and upload data for the website's trajectory/response visualization:

```bash
python scripts/upload_experiment_bundle.py --visualization
```

This adds `trajectory` to each episode (every interpolated commanded action, including XYZ, rot6d, gripper, and zero-based simulator step). Each `calls` entry gains `step`, `observed_xyz`, `video_time_s`, `tools` (names and full target arguments/notes), and `response_text`. Response markers use the observation preceding the assistant response; `observed_xyz` is the measured position rounded in the original transcript. Tool targets are requested targets, while `trajectory` contains executed commanded targets, not measured robot motion. Missing observations are null; do not invent a marker for them. No images or video bytes are exported. Saved API response text and tool notes are included only with this flag.

Video time is simulator step / 20, at the current recording's 20 fps. Videos include an initial reset frame, so the video frame after action step `t` is at `(t + 1) / 20`. Existing Drive previews are included under each episode's `video` when listed in the local uploaded-video index. Drive preview iframes do not support reliable programmatic seeking; show the timestamp alongside the marker.

Copy `scripts/fetch_experiment_results.py` into the other repo and install `google-api-python-client google-auth-oauthlib`. Run it as a local/build/server task, using the file ID printed by the uploader:

```bash
python scripts/fetch_experiment_results.py --file-id DRIVE_FILE_ID \
  --token /private/path/token.json \
  --output public/data/experiment_results.json
```

Keep the OAuth token outside public assets. Publishing the downloaded JSON makes response text and notes visible to visitors. Rerun the uploader and fetcher to refresh the snapshot.

In the website:

```javascript
const results = await fetch('/data/experiment_results.json').then(r => r.json());
const episode = results.episodes.find(e => e.model === selectedModel &&
  e.effort === selectedEffort && e.task_id === selectedTask && e.init_state === selectedState);
const path = episode?.trajectory ?? [];
const markers = (episode?.calls ?? []).filter(c => c.observed_xyz != null);
// Draw path[].xyz as the commanded polyline; markers[].observed_xyz as response dots.
// Clicking a dot displays its tools, response_text, token counts, cost, and video_time_s.
```

# Rollout videos for the website

Two independent scripts: fetch copies only MP4s from metadisk; upload sends only those MP4s to Google Drive. Both accept `--model`, `--reasoning` (alias `--effort`), and `--task` (alias `--task-number`). Each defaults to `all`. Model names can include the provider or omit it. Task numbers use the experiment's **zero-based 0–9 IDs**. When a campaign manifest is present, superseded runs and individual states replaced by completed retries are excluded, matching the results export. Without a manifest, all available runs are included, with distinct keys.

## Fetch (no extra dependencies)

Run from the repository root:

```bash
python scripts/fetch_rollout_videos.py --model all --reasoning all --task all --dry-run
python scripts/fetch_rollout_videos.py --model all --reasoning all --task all
# Example subset:
python scripts/fetch_rollout_videos.py --model gpt-6.1-sol --reasoning low --task 2
```

Default source: `/metadisk/may/llm_libero_experiments/libero_experiments_20260930`.
Override with `--source /metadisk/path/to/campaign`. Default destination: `outputs/website_videos`; override with `--output`. Summaries are read for metadata, but only videos are copied. The script resolves old workspace video paths into the corresponding episode folders on metadisk. Missing videos are reported and skipped. Sources are never modified. Each fetch writes `videos.json` for the current selection; previous copied videos remain on disk but are not selected for upload unless listed in that index.

## Google Drive setup and upload

Follow Google's [Python quickstart](https://developers.google.com/workspace/drive/api/quickstart/python) to enable the Drive API, configure OAuth consent (add your account as a test user for an external testing app), and download a **Desktop app** OAuth client JSON. Store it at `outputs/drive_auth/credentials.json`.

```bash
python -m pip install google-api-python-client google-auth-oauthlib
python scripts/upload_rollout_videos.py --model all --reasoning all --task all --dry-run
python scripts/upload_rollout_videos.py --credentials outputs/drive_auth/credentials.json --public
```

Each collaborator should keep their own OAuth credentials and token. To add a
partner's videos to the same folder, give them access to the folder and have
them run the fetch command on their campaign, followed by the upload command
with the shared `--folder-id`. Identical videos are skipped; new rollout keys
are added without deleting other collaborators' files.

The default destination is folder `1D6vn9HiG9V81ROdV57m7zy8tOsDLL84M`. Sign in as an account that can add files there. OAuth prints a login URL; open it in a browser on the same machine so the localhost callback can complete. On a remote machine, forward the callback port printed in the login URL. The script requests full Drive scope to access your existing folder by ID without a Google Picker grant; see Google's [scope documentation](https://developers.google.com/workspace/drive/api/guides/api-specific-auth). The code uploads only selected videos to the destination folder. Credentials and tokens stay under ignored `outputs/`.

`--public` grants anyone with the link reader access to selected videos. Omit it to retain existing/inherited sharing. Public website visitors need viewer access; an organization's sharing policy may prevent public access. See Google's [sharing documentation](https://developers.google.com/workspace/drive/api/guides/manage-sharing).

Videos use readable filenames such as `openai_gpt-6.1-sol_low_task_02_state_03_success.mp4`. The outcome suffix is `success`, `fail`, or `unknown` when no result is recorded. Multiple rollouts with the same labels get a `_run_01` suffix. Rerunning the uploader renames previously uploaded hash filenames while keeping their Drive IDs and links. Existing local indexes with hash filenames still work.

Uploads use resumable transfers with retries. Re-running finds existing videos by a stable key, skips identical content, and updates changed content. The index is saved after every completed video so an interrupted batch can be rerun. `--folder-id`, `--input`, and `--token` override defaults. Filters work independently at upload time.

## Website integration

Copy `outputs/website_videos/drive_videos.json` into your website's public assets. Its `videos` array includes model, reasoning, task, initial state, outcome, cost, Drive file ID, and preview/view URLs. Use these records to populate the rollout picker. Only completed uploads appear in this index; it is not the full experiment results dataset.

Display the selected record's `preview_url` in an iframe:

```html
<iframe src="https://drive.google.com/file/d/FILE_ID/preview"
        title="Selected rollout" width="960" height="540"
        allow="fullscreen" allowfullscreen></iframe>
```

Use the Drive preview player rather than treating the share URL as an MP4 URL for an HTML `<video>` element. Check playback while signed out once Drive finishes processing the videos.

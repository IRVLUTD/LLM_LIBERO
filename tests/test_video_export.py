"""Exercise selection, relocated paths, and the fetch/upload boundary."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'fetch_rollout_videos.py'
spec = importlib.util.spec_from_file_location('fetch_rollout_videos', SCRIPT)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def test_relocated_videos_and_selectors(tmp_path):
    campaign = tmp_path / 'campaign'
    run = campaign / 'model_low' / 'task_02'
    episode = run / 'episode_00' / 'recording'
    episode.mkdir(parents=True)
    (episode / 'rollout.mp4').write_bytes(b'video')
    (episode / 'private_log.json').write_text('secret')
    (run / 'summary.json').write_text(json.dumps({
        'protocol': {'model': 'openai/example', 'reasoning_effort': 'low', 'task_id': 2, 'initial_state_ids': [3]},
        'results': [{'episode': 0, 'video': '/old/workspace/campaign/model_low/task_02/episode_00/recording/rollout.mp4', 'environment_success': False}],
    }))
    video, row = list(fetch.discover(campaign))[0]
    assert video == episode / 'rollout.mp4'
    assert row['init_state'] == 3 and row['success'] is False
    assert fetch.matches(row, SimpleNamespace(model='example', reasoning='low', task='2'))
    assert fetch.matches(row, SimpleNamespace(model='all', reasoning='all', task='all'))
    assert not fetch.matches(row, SimpleNamespace(model='example', reasoning='high', task='2'))
    output = tmp_path / 'export'
    subprocess.run([sys.executable, str(SCRIPT), '--source', str(campaign), '--output', str(output)], check=True)
    assert (output / row['file']).read_bytes() == b'video'
    assert not list(output.rglob('private_log.json'))
    index = json.loads((output / 'videos.json').read_text())
    assert index['videos'] == [row]
    # Upload dry-run works without credentials or installed Google libraries.
    subprocess.run([sys.executable, str(SCRIPT.with_name('upload_rollout_videos.py')), '--input', str(output), '--dry-run'], check=True)


def test_outside_symlink_is_skipped(tmp_path):
    campaign = tmp_path / 'campaign'
    run = campaign / 'run'
    run.mkdir(parents=True)
    outside = tmp_path / 'outside.mp4'
    outside.write_bytes(b'private')
    (run / 'rollout.mp4').symlink_to(outside)
    (run / 'summary.json').write_text(json.dumps({'protocol': {'model': 'example', 'reasoning_effort': 'low', 'task_id': 0}, 'results': [{'video': 'rollout.mp4'}]}))
    assert list(fetch.discover(campaign)) == []


def test_manifest_excludes_replaced_states_and_cancelled_runs(tmp_path):
    campaign = tmp_path / 'campaign'
    manifest = {'runs': [
        {'directory': 'original', 'status': 'finished', 'superseded_initial_state_ids': [0]},
        {'directory': 'retry', 'status': 'finished'},
        {'directory': 'cancelled', 'status': 'superseded'}]}
    campaign.mkdir()
    (campaign / 'campaign.json').write_text(json.dumps(manifest))
    for name, states in [('original', [0, 1]), ('retry', [0]), ('cancelled', [2])]:
        run = campaign / name
        results = []
        for ep, state in enumerate(states):
            video = run / f'episode_{ep:02d}' / 'rollout.mp4'
            video.parent.mkdir(parents=True)
            video.write_bytes(b'video')
            results.append({'episode': ep, 'video': str(video)})
        (run / 'summary.json').write_text(json.dumps({'protocol': {
            'model': 'example', 'reasoning_effort': 'medium', 'task_id': 2,
            'initial_state_ids': states}, 'results': results}))
    rows = list(fetch.discover(campaign))
    assert [(row['source_relative'].split('/')[0], row['init_state']) for _, row in rows] == [
        ('original', 1), ('retry', 0)]

"""Results export keeps numerical evidence and distinguishes unfinished episodes."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import upload_experiment_bundle as bundle


def test_merge_concatenates_and_deduplicates_exact_episode():
    old = {'version': 1, 'campaign': 'old', 'episodes': [
        {'model': 'a', 'effort': 'low', 'task_id': 0, 'init_state': 0,
         'run': 'one', 'episode': 0, 'has_result': True, 'success': False}]}
    new = {'version': 1, 'campaign': 'new', 'created_at': 'now', 'episodes': [
        {'model': 'a', 'effort': 'low', 'task_id': 0, 'init_state': 0,
         'run': 'one', 'episode': 0, 'has_result': True, 'success': True},
        {'model': 'b', 'effort': 'medium', 'task_id': 1, 'init_state': 2,
         'run': 'two', 'episode': 0, 'has_result': True, 'success': False}]}
    merged = bundle.merge_results(old, new)
    assert len(merged['episodes']) == 2
    assert merged['episodes'][0]['success'] is True
    assert merged['totals']['successes'] == 1
    assert merged['merged_sources'] == ['new', 'old']


def test_numbers_only_and_pending(tmp_path):
    run = tmp_path / 'run'
    (run / 'api').mkdir(parents=True)
    (tmp_path / 'campaign.json').write_text(json.dumps({'runs': [
        {'directory': 'run', 'model': 'example', 'effort': 'medium', 'task_id': 0, 'initial_state_ids': [0, 1, 2]},
        {'directory': 'old', 'status': 'superseded'}]}))
    (run / 'summary.json').write_text(json.dumps({'results': [
        {'episode': 0, 'environment_success': True, 'status': 'success', 'steps': 20, 'estimated_usd': 0.5, 'video': '/secret/video.mp4'},
        {'episode': 1, 'status': 'interrupted', 'environment_success': None}]}))
    event = {'event': 'response', 'episode': 0, 'call': 1, 'usage': {'input_tokens': 10, 'output_tokens': 5, 'output_tokens_details': {'reasoning_tokens': 3}}, 'latency_s': 2, 'estimated_usd': 0.5}
    (run / 'api/api_usage.jsonl').write_text(json.dumps(event) + '\n' + json.dumps(event) + '\n{"unfinished":')
    # Large evidence must never be read by the numerical exporter.
    (run / 'api/ep00_call01_response.json').write_text('invalid raw response')
    data = bundle.export_results(tmp_path)
    assert data['totals']['scheduled'] == 3
    assert data['totals']['successes'] == 1
    assert data['totals']['failures'] == 0
    assert data['totals']['unknown_outcomes'] == 1
    assert data['totals']['pending_or_missing'] == 1
    assert data['totals']['input_tokens'] == 10
    assert data['totals']['reasoning_tokens'] == 3
    assert data['episodes'][2]['output_tokens'] is None
    assert '/secret' not in json.dumps(data)
    assert 'conversation' not in json.dumps(data)
    assert len(data['by_model_task']) == 1


def test_response_marker_and_interpolated_path(tmp_path):
    from response_visualization import visualization
    root = tmp_path / 'episode_00' / 'inspect'
    (root / 'transcripts').mkdir(parents=True)
    (root / 'actions').mkdir()
    (tmp_path / 'api').mkdir()
    messages = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Current observation.\nstate[eef_state]: x=0.1 y=0.2 z=0.3 gripper=0\ncamera (step 7):'}]},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'function': {'name': 'move_to'}}]}]
    (root / 'transcripts/task00_init00.jsonl').write_text('\n'.join(json.dumps(m) for m in messages))
    (root / 'actions/task00_init00.jsonl').write_text(json.dumps({'t': 7, 'action': [0.2, 0.3, 0.4, 1, 0, 0, 0, 1, 0, 1]}))
    (tmp_path / 'api/ep00_call01_response.json').write_text(json.dumps({'output': [{'type': 'function_call', 'name': 'move_to', 'arguments': '{"targets":{"x":0.5},"note":"Move above object"}'}]}))
    calls = [{'call': 1}, {'call': 2}]
    path = visualization(tmp_path, 0, 0, calls)
    assert calls[0]['step'] == 7
    assert calls[0]['observed_xyz'] == [0.1, 0.2, 0.3]
    assert calls[0]['video_time_s'] == 0.35
    assert calls[0]['tools'][0]['arguments']['targets']['x'] == 0.5
    assert calls[1]['step'] is None
    assert path[0]['xyz'] == [0.2, 0.3, 0.4]


def test_export_replaces_only_superseded_initial_states(tmp_path):
    common = {'model': 'example', 'effort': 'medium', 'task_id': 2, 'status': 'finished'}
    (tmp_path / 'campaign.json').write_text(json.dumps({'runs': [
        dict(common, directory='original', initial_state_ids=[0, 1], superseded_initial_state_ids=[0]),
        dict(common, directory='retry', initial_state_ids=[0])]}))
    data = bundle.export_results(tmp_path)
    assert [(row['run'], row['init_state']) for row in data['episodes']] == [('original', 1), ('retry', 0)]
    assert data['totals']['scheduled'] == 2

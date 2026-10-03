"""Copy selected rollout videos from experiment storage and write a website index."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil

DEFAULT_SOURCE = Path('/metadisk/may/llm_libero_experiments/libero_experiments_20260930')


def add_filters(parser):
    parser.add_argument('--model', default='all', help='Full model name or name without provider; default: all')
    parser.add_argument('--reasoning', '--effort', default='all', help='Reasoning level; default: all')
    parser.add_argument('--task', '--task-number', default='all', help='Zero-based task number (0–9); default: all')


def matches(row, args):
    return (args.model == 'all' or args.model in (row['model'], row['model'].split('/')[-1])) and (
        args.reasoning == 'all' or args.reasoning == row['reasoning']) and (
        args.task == 'all' or int(args.task) == row['task_id'])


def video_name(row):
    model = re.sub(r'[^A-Za-z0-9._-]+', '_', row['model']).strip('_')
    reasoning = re.sub(r'[^A-Za-z0-9._-]+', '_', row['reasoning']).strip('_')
    state = row.get('init_state')
    state_label = f'{state:02d}' if isinstance(state, int) else 'unknown'
    outcome = 'success' if row.get('success') is True else 'fail' if row.get('success') is False else 'unknown'
    return f'{model}_{reasoning}_task_{row["task_id"]:02d}_state_{state_label}_{outcome}.mp4'


def video_names(rows):
    """Add readable run numbers only when multiple rollouts share the same labels."""
    groups = {}
    for row in rows:
        groups.setdefault(video_name(row), []).append(row)
    names = {}
    for name, group in groups.items():
        for number, row in enumerate(sorted(group, key=lambda r: r['source_relative']), 1):
            names[row['key']] = name if len(group) == 1 else f'{name[:-4]}_run_{number:02d}.mp4'
    return names


def write_index(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps({'version': 1, 'videos': rows}, indent=2) + '\n')
    temporary.replace(path)


def discover(source):
    source = source.resolve()
    manifest_path = source / 'campaign.json'
    runs = ({run['directory']: run for run in json.loads(manifest_path.read_text()).get('runs', [])}
            if manifest_path.is_file() else None)
    for summary_path in sorted(source.rglob('summary.json')):
        run = runs.get(summary_path.parent.relative_to(source).as_posix()) if runs is not None else None
        if runs is not None and (run is None or run.get('status') == 'superseded'):
            continue
        summary = json.loads(summary_path.read_text())
        protocol = summary.get('protocol') or json.loads((summary_path.parent / 'protocol.json').read_text())
        for result in summary.get('results', []):
            state = (protocol.get('initial_state_ids') or [None])[result.get('episode', 0)]
            if run and state in run.get('superseded_initial_state_ids', []):
                continue
            if not result.get('video'):
                continue
            stored = Path(result['video'])
            # Episode directories retain their names when a campaign is moved.
            parts = stored.parts
            episode_parts = next((parts[i:] for i, part in enumerate(parts) if part.startswith('episode_')), None)
            video = summary_path.parent.joinpath(*episode_parts) if episode_parts else summary_path.parent / stored.name
            if not video.is_file() or not video.resolve().is_relative_to(source):
                print(f'Skipping missing/outside video: {video}')
                continue
            relative = video.resolve().relative_to(source).as_posix()
            key = hashlib.sha256((source.name + '/' + relative).encode()).hexdigest()
            yield video, {
                'key': key, 'model': protocol['model'], 'reasoning': protocol['reasoning_effort'],
                'task_id': protocol['task_id'], 'task_name': protocol.get('task_name'),
                'instruction': protocol.get('instruction'), 'episode': result.get('episode'),
                'init_state': (protocol.get('initial_state_ids') or [None])[result.get('episode', 0)],
                'status': result.get('status'), 'success': result.get('environment_success'),
                'estimated_usd': result.get('estimated_usd'), 'source_relative': relative,
                'file': 'videos/' + video_name({'model': protocol['model'], 'reasoning': protocol['reasoning_effort'], 'task_id': protocol['task_id'], 'init_state': (protocol.get('initial_state_ids') or [None])[result.get('episode', 0)], 'success': result.get('environment_success')}), 'size_bytes': video.stat().st_size,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=Path('outputs/website_videos'))
    parser.add_argument('--dry-run', action='store_true')
    add_filters(parser)
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f'Source directory does not exist: {args.source}')
    if args.task != 'all' and (not args.task.isdigit() or not 0 <= int(args.task) <= 9):
        parser.error('--task must be all or 0–9')
    selected = [(video, row) for video, row in discover(args.source) if matches(row, args)]
    if not selected:
        parser.error('No videos match these selectors')
    print(f'{len(selected)} videos, {sum(row["size_bytes"] for _, row in selected) / 1024**3:.2f} GiB')
    if args.dry_run:
        for _, row in selected:
            print(f'{row["model"]} {row["reasoning"]} task={row["task_id"]} state={row["init_state"]}: {row["source_relative"]}')
        return
    names = video_names([row for _, row in selected])
    rows = []
    for video, row in selected:
        row['file'] = 'videos/' + names[row['key']]
        target = args.output / row['file']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != video.stat().st_size:
            shutil.copyfile(video, target)
        rows.append(row)
    write_index(args.output / 'videos.json', rows)
    print(f'Index: {args.output / "videos.json"}')


if __name__ == '__main__':
    main()

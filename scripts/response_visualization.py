"""Compact response markers and commanded trajectories from saved rollout evidence."""
import json
from pathlib import Path
import re


def lines(path):
    with path.open() as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def visualization(folder, episode, state, calls):
    root = folder / f'episode_{episode:02d}' / 'inspect'
    observations = []
    for path in sorted((root / 'transcripts').rglob('*.jsonl')):
        if f'init{state:02d}' not in path.name:
            continue
        context = {'step': None, 'observed_xyz': None}
        for message in lines(path):
            content = message.get('content')
            text = content if isinstance(content, str) else '\n'.join(p.get('text', '') for p in content or [] if isinstance(p, dict))
            if message.get('role') == 'user' and 'Current observation.' in text:
                step = re.search(r'\(step (\d+)\)', text)
                position = re.search(r'state\[eef_state\]: x=([^ ]+) y=([^ ]+) z=([^ \n]+)', text)
                context = {'step': int(step[1]) if step else None,
                           'observed_xyz': [float(position[i]) for i in range(1, 4)] if position else None}
            if message.get('role') == 'assistant':
                observations.append(dict(context))
    trajectory = []
    for path in sorted((root / 'actions').rglob('*.jsonl')):
        if f'init{state:02d}' not in path.name:
            continue
        for row in lines(path):
            action = row.get('action')
            if isinstance(action, list) and len(action) == 10:
                trajectory.append({'step': row['t'], 'xyz': action[:3], 'rotation_6d': action[3:9], 'gripper': action[9]})
    for call in calls:
        number = call['call']
        if not isinstance(number, int):
            continue
        context = observations[number - 1] if 0 < number <= len(observations) else {'step': None, 'observed_xyz': None}
        call.update(context, video_time_s=context['step'] / 20 if context['step'] is not None else None)
        path = folder / 'api' / f'ep{episode:02d}_call{number:02d}_response.json'
        payload = json.loads(path.read_text()) if path.is_file() else {}
        tools, texts = [], []
        items = payload.get('output', []) + payload.get('content', [])
        # Chat Completions responses (Gemini) carry one message with tool_calls.
        for choice in payload.get('choices', []):
            message = choice.get('message') or {}
            if message.get('content'):
                items.append({'type': 'text', 'text': message['content']})
            items.extend({'type': 'function_call', **call.get('function', {})} for call in message.get('tool_calls') or [])
        for item in items:
            if item.get('type') in {'function_call', 'tool_use'}:
                args = item.get('input', item.get('arguments', {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {'unparsed': args}
                tools.append({'name': item.get('name'), 'arguments': args})
            if item.get('type') == 'text':
                texts.append(item.get('text', ''))
            if item.get('type') == 'message':
                texts.extend(p.get('text', '') for p in item.get('content', []) if p.get('type') in {'text', 'output_text'})
        call.update(tools=tools, response_text='\n'.join(texts))
    return trajectory

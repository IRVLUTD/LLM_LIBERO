"""Demonstration storyboards as one extra user message (few-shot visual context).

Storyboards come from ``scripts/prepare_demonstrations.py``. The manifest pins
each PNG by SHA-256 so a run's demonstrations are auditable.
"""
import base64
import hashlib
import json
from pathlib import Path


def _image_block(b64, wire):
    if wire=='responses':
        return {'type':'input_image','image_url':'data:image/png;base64,'+b64,'detail':'high'}
    return {'type':'image','source':{'type':'base64','media_type':'image/png','data':b64}}


def _load_keyframes(manifest, task_id, wire):
    """Hand-picked key frames: an intro, then each image after its own caption, then an outro."""
    if manifest['task_id']!=task_id or not manifest['frames']:
        raise ValueError('Expected at least one key frame of the requested task')
    text_type='input_text' if wire=='responses' else 'text'
    content=[{'type':text_type,'text':manifest['intro']}]
    datas=[]
    for frame in manifest['frames']:
        content.append({'type':text_type,'text':frame['caption']})
        # A frame has one image, or several labelled views (e.g. external and wrist camera).
        views=frame.get('images') or [{'image':frame['image'],'sha256':frame['sha256']}]
        for view in views:
            raw=Path(view['image']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=view['sha256']:
                raise ValueError(f'Key frame checksum mismatch: {view["image"]}')
            b64=base64.b64encode(raw).decode()
            datas.append(b64)
            if view.get('label'):
                content.append({'type':text_type,'text':view['label']})
            content.append(_image_block(b64, wire))
    if manifest.get('outro'):
        content.append({'type':text_type,'text':manifest['outro']})
    return manifest, {'role':'user','content':content}, frozenset(datas)


def load_demonstrations(path, task_id, wire='responses'):
    manifest=json.loads(Path(path).read_text())
    if wire not in ('responses','messages'):
        raise ValueError(f'unknown wire {wire!r}')
    if manifest.get('kind')=='keyframes':
        return _load_keyframes(manifest, task_id, wire)
    n=len(manifest['demos'])
    if manifest['task_id']!=task_id or n<1:
        raise ValueError('Expected at least one demonstration of the requested task')
    if len({d['demo_id'] for d in manifest['demos']})!=n:
        raise ValueError('Demonstrations must be distinct')
    if wire not in ('responses','messages'):
        raise ValueError(f'unknown wire {wire!r}')
    word={1:'one',2:'two',3:'three',4:'four',5:'five'}.get(n,str(n))
    intro=(f'Here are {word} recorded demonstration{"s" if n!=1 else ""} of this task, provided as visual examples. '
        'Each image is one demonstration: read its 12 tiles left-to-right, then top-to-bottom. '
        'Inside each tile, the external camera is on the left and wrist camera on the right. '
        'Labels indicate chronological frame/time. These are uniformly sampled frames, not full videos. '
        'Use them to understand the intended object interactions and final arrangement. '
        'They are NOT live observations. Object placements in the live trial may differ. '
        'Use the subsequent live observations and robot state to control this episode. '
        'No action labels or object coordinates are supplied.')
    text_type='input_text' if wire=='responses' else 'text'
    content=[{'type':text_type,'text':intro}]
    datas=[]
    for demo in manifest['demos']:
        raw=Path(demo['storyboard']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=demo['storyboard_sha256']:
            raise ValueError('Storyboard checksum mismatch')
        b64=base64.b64encode(raw).decode()
        datas.append(b64)
        content.append({'type':text_type,'text':f'Recorded demonstration {demo["demo_id"]}.'})
        if wire=='responses':
            content.append({'type':'input_image','image_url':'data:image/png;base64,'+b64,'detail':'high'})
        else:
            content.append({'type':'image','source':{'type':'base64','media_type':'image/png','data':b64}})
    return manifest, {'role':'user','content':content}, frozenset(datas)

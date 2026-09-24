"""Simulator-only check: coordinate frames, tool interpolation, gripper direction.

No LLM and no API key. Lifts the end effector 3 cm with the agent's own
``move_to`` tool, then closes and opens the gripper, and asserts LIBERO tracked
each target. Needs LIBERO and EGL.

    python scripts/check_adapter.py
"""
import json
import os
from pathlib import Path

import numpy as np
from inspect_robots._dotenv import read_dotenv
from inspect_robots.scene import Scene
from inspect_robots_agent._tools import build_toolset
from inspect_robots_agent._llm import ToolCall

from inspect_robots_libero.embodiment import LiberoEmbodiment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'checks' / 'adapter'
settings = {**read_dotenv(ROOT / '.env'), **os.environ}

env = LiberoEmbodiment(task_id=0, output_dir=OUT, libero_python=settings.get('LIBERO_PYTHON'))
try:
    obs = env.reset(Scene(id='offline', instruction='offline adapter check'))
    toolset = build_toolset(env.info.action_space, env.info.observation_space, control_hz=20)
    start = obs.state['eef_state'].copy()
    target = float(start[2]+.03)
    result = toolset.execute(ToolCall(id='offline', name='move_to', arguments=json.dumps(
        {'targets':{'z':target}, 'note':'Offline vertical lift to verify coordinate conversion.'})), obs)
    assert result.error is None, result.error
    for action in result.chunk.actions:
        obs = env.step(action).observation
    # Hold the target to assess tracking without new policy requests.
    for _ in range(20):
        obs = env.step(result.chunk.actions[-1]).observation
    end = obs.state['eef_state'].copy()
    assert abs(end[2]-target)<.012, (start.tolist(),end.tolist(),target)
    assert np.linalg.norm(end[:2]-start[:2])<.015
    assert np.linalg.norm(end[3:9]-start[3:9])<.12, (start.tolist(), end.tolist())
    grip_values=[]
    for grip in (1.,0.):
        r=toolset.execute(ToolCall(id='grip',name='move_to',arguments=json.dumps(
            {'targets':{'gripper':grip},'note':'Offline gripper direction check.'})),obs)
        assert r.error is None, r.error
        for a in r.chunk.actions:
            obs=env.step(a).observation
        for _ in range(20):
            obs=env.step(r.chunk.actions[-1]).observation
        grip_values.append(float(obs.state['eef_state'][-1]))
    assert grip_values[0]>.7 and grip_values[1]<.15, grip_values
    report={'gripper_closed_then_open':grip_values,'start':start.tolist(),'end':end.tolist(),'target_z':target,
        'interpolation_steps':len(result.chunk.actions),'api_calls':0,'passed':True}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
finally:
    env.close()

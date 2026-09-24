"""Private stdio LIBERO worker, launched by ``LiberoEmbodiment`` as a subprocess.

Does not import ``inspect_robots``, so it can also run under a separate
``$LIBERO_PYTHON`` interpreter. Protocol: one JSON request per stdin line
(``reset`` / ``step`` / ``close``), one JSON reply per stdout line. No
credentials or network model clients.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

# Imports and LIBERO diagnostics go to stderr; stdout is exclusively protocol JSON.
wire = sys.stdout
sys.stdout = sys.stderr
# LIBERO's first import asks an interactive question when its config file is
# missing; here stdin is the protocol pipe, so fail loudly instead of hanging.
_config = os.path.join(os.environ.get('LIBERO_CONFIG_PATH', os.path.expanduser('~/.libero')),
                       'config.yaml')
if not os.path.exists(_config):
    wire.write(json.dumps({'error': f'LIBERO is not initialised ({_config} missing). Run '
                           '"python -c \'import libero.libero\'" once in a terminal '
                           'and answer N (see README, Installation).'}) + '\n')
    wire.flush()
    sys.exit(1)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.transform_utils as T
import imageio.v2 as imageio


def main():
    env = None
    video = None
    obs = None
    steps = 0
    def pack():
        return {
            'images': {k: (obs[k][::-1, ::-1] if k == 'agentview_image' else obs[k]).copy().tolist() for k in
                       ('agentview_image', 'robot0_eye_in_hand_image')},
            'eef_state': np.concatenate((env.env.robots[0].controller.ee_pos,
                env.env.robots[0].controller.ee_ori_mat[:, :2].T.reshape(-1),
                [np.clip(1.0 - np.sum(np.abs(obs['robot0_gripper_qpos'])) / .08, 0, 1)])).tolist(),
            'gripper_qpos': obs['robot0_gripper_qpos'].tolist(),
            'success': bool(env.check_success()), 'steps': steps,
        }
    try:
        for line in sys.stdin:
            req = json.loads(line)
            try:
                if req['op'] == 'reset':
                    if env is not None:
                        env.close()
                    if video is not None:
                        video.close()
                    suite = benchmark.get_benchmark_dict()['libero_10']()
                    task_id = int(req['task_id'])
                    task = suite.get_task(task_id)
                    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) /
                        task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
                    init_state = int(req['init_state'])
                    env.seed(int(req['env_seed']))
                    env.reset()
                    obs = env.set_init_state(suite.get_task_init_states(task_id)[init_state])
                    for _ in range(10):
                        obs, _, _, _ = env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
                    env.env.robots[0].controller.use_delta = False
                    steps = 0
                    video = imageio.get_writer(req['video'], fps=20)
                    result = pack()
                    result['instruction'] = task.language
                elif req['op'] == 'step':
                    action = np.asarray(req['action'], dtype=float)
                    if action.shape != (10,) or not np.all(np.isfinite(action)):
                        raise ValueError('Invalid action')
                    # Upstream uses rot6d; orthonormalize its two column vectors.
                    a = action[3:6]
                    b = action[6:9]
                    if np.linalg.norm(a) < 1e-6:
                        raise ValueError('Degenerate rot6d first column')
                    a = a / np.linalg.norm(a)
                    b = b - np.dot(a, b) * a
                    if np.linalg.norm(b) < 1e-6:
                        raise ValueError('Degenerate rot6d second column')
                    b = b / np.linalg.norm(b)
                    rotation = np.stack([a, b, np.cross(a, b)], axis=1)
                    action = np.concatenate([action[:3], T.quat2axisangle(T.mat2quat(rotation)),
                                             [2 * action[-1] - 1]])
                    obs, _, _, _ = env.step(action)
                    steps += 1
                    result = pack()
                elif req['op'] == 'close':
                    break
                else:
                    raise ValueError('Unknown operation')
                video.append_data(np.concatenate([np.asarray(result['images'][k], dtype=np.uint8)
                    for k in ('agentview_image', 'robot0_eye_in_hand_image')], axis=1))
                wire.write(json.dumps(result) + '\n')
                wire.flush()
            except Exception as exc:
                wire.write(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)}) + '\n')
                wire.flush()
    finally:
        if video is not None:
            video.close()
        if env is not None:
            env.close()

if __name__ == '__main__':
    main()

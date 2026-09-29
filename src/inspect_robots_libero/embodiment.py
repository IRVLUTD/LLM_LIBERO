"""LIBERO-10 embodiment: a Panda arm in MuJoCo, driven over a private stdio worker.

The simulator runs in a subprocess (``_worker.py``, by default under the same
interpreter) and this class speaks line-delimited JSON with it. This keeps API
credentials out of the simulator process and isolates MuJoCo/EGL crashes from
the eval loop.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import IO, Any

import numpy as np

from inspect_robots.embodiment import EmbodimentBase, EmbodimentInfo
from inspect_robots.errors import ConfigError, EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_libero.notes import TASK_NOTES
from inspect_robots_libero.tasks import ENV_SEED_BASE, check_task_id

CONTROL_HZ = 20
WORKER = Path(__file__).with_name("_worker.py")

BASE_DOCS = (
    "Single Panda arm. move_to uses world-frame x,y,z in meters and rot6d "
    "orientation (first two rotation-matrix columns). z points up. Gripper 0=open, 1=close; "
    "measured closure may remain below 1 while holding an object. Keep orientation "
    "unchanged unless necessary. Images are upright agentview and wrist camera views. "
    "Only robot proprioception and RGB are available; no object coordinates. "
    "Intermediate target arrival must be checked from the next observation. "
    "IMPORTANT: omitting gripper from a move_to call holds it at the CURRENT measured "
    'closure, not "stay closed" -- a grip that has already slipped partly open will stay '
    "that way or loosen further. While transporting or holding an object, explicitly "
    "include gripper=1 in every move_to call, not only when first closing it, and verify "
    "the object actually followed the lift in the next observation before assuming the "
    "grasp is secure.\n\n"
    "General manipulation guidance:\n"
    "1. Avoid collisions: before any lateral move, lift above all nearby objects and "
    "furniture. Do not sweep through or push objects unless the task requires it. Use "
    "large moves in free space and small moves only in the last few centimeters near an "
    "object.\n"
    "2. Identify the target object carefully; similar-looking distractors may be present. "
    "Cross-check both camera views.\n"
    "3. A closed gripper does not mean a successful grasp. After closing, lift 3-5 cm and "
    "confirm in both cameras that the object rose with the gripper.\n"
    "4. Handle objects the way a person would: grasp a stable, graspable part without "
    "pushing or tipping the object, keep it in the same orientation (usually upright) "
    "from grasp to placement, and lower it close to the support surface before "
    "releasing.\n"
    "5. If the same approach fails twice, change it (grasp point, approach direction, or "
    "a small adjustment); do not repeat the same motion.\n"
    "6. After each move, confirm from the next observation that it succeeded before "
    "starting the next step; do not assume success.\n"
    "7. Prefer fewer, purposeful moves over many tiny corrections; your call budget is "
    "limited."
)


def _action_space() -> Box:
    """Absolute end-effector pose (xyz + rot6d) plus a continuous gripper command."""
    return Box(
        shape=(10,),
        # Living-room tables sit at z=0.41, where the fingertips touch the table with the
        # end effector at about 0.45; the floor stays just below that so flat objects are
        # reachable without letting targets drive hard into the table.
        low=np.array([-0.6, -0.6, 0.42, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
        high=np.array([0.6, 0.6, 1.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        semantics=ActionSemantics(
            control_mode="eef_abs_pose",
            rotation_repr="rot6d",
            gripper="continuous",
            frame="world",
            dim_labels=("x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper"),
            max_step=(0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1),
        ),
    )


class LiberoEmbodiment(EmbodimentBase):
    """One LIBERO-10 simulator instance; each reset opens a fresh episode.

    Args:
        task_id: native LIBERO-10 task id. Optional when scenes come from the
            ``libero_10`` task (their metadata names the task); required for
            ``--instruction`` runs and for ``task_notes``.
        init_state: initial-state index used when a scene carries none.
        output_dir: where ``worker.log`` and one ``rollout.mp4`` per episode go.
        libero_python: interpreter that runs the worker. Defaults to
            ``$LIBERO_PYTHON``, else the current interpreter.
        task_notes: append the demonstration-derived hints in ``notes.py``
            (not zero-shot; off by default).
    """

    def __init__(
        self,
        task_id: int | None = None,
        init_state: int = 0,
        output_dir: str | os.PathLike[str] = "outputs/libero",
        libero_python: str | None = None,
        task_notes: bool = False,
    ) -> None:
        self.task_id = None if task_id is None else check_task_id(task_id)
        self.init_state = init_state
        self.output = Path(output_dir)
        self.libero_python = libero_python or os.environ.get("LIBERO_PYTHON") or sys.executable
        if task_notes and self.task_id is None:
            raise ConfigError("task_notes=True needs an explicit task_id.\nfix: add -E task_id=N")
        docs = BASE_DOCS
        if task_notes and self.task_id in TASK_NOTES:
            docs += "\n\n" + TASK_NOTES[self.task_id]
        name = "libero_10" if self.task_id is None else f"libero_10_task{self.task_id:02d}"
        self.info = EmbodimentInfo(
            name=name,
            is_simulated=True,
            control_hz=CONTROL_HZ,
            capabilities=frozenset({"seedable", "resettable", "privileged_success", "renderable"}),
            action_space=_action_space(),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(k, 256, 256) for k in ("agentview", "wrist")),
                state=StateSpec(
                    (StateField("eef_state", (10,)), StateField("gripper_qpos", (2,), "m"))
                ),
            ),
            docs=docs,
        )
        self.process: subprocess.Popen[str] | None = None
        self.worker_log: IO[str] | None = None
        self.last: dict[str, Any] | None = None
        self.instruction = ""
        self.episode_dir: Path | None = None

    def _rpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return the worker's reply; worker errors raise."""
        assert self.process is not None and self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise EmbodimentFault(f"LIBERO worker exited; see {self.output / 'worker.log'}")
        result: dict[str, Any] = json.loads(line)
        if "error" in result:
            raise EmbodimentFault(f"LIBERO worker: {result['error']}")
        self.last = result
        return result

    def _observation(self, data: dict[str, Any]) -> Observation:
        images = {
            name: np.asarray(data["images"][key], dtype=np.uint8)
            for name, key in (("agentview", "agentview_image"), ("wrist", "robot0_eye_in_hand_image"))
        }
        state = {key: np.asarray(data[key], dtype=float) for key in ("eef_state", "gripper_qpos")}
        return Observation(
            images=images,
            state=state,
            instruction=self.instruction,
            state_time=data["steps"] / CONTROL_HZ,
        )

    def _start_worker(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        env = {
            key: value
            for key, value in os.environ.items()
            if "API_KEY" not in key and "TOKEN" not in key and "SECRET" not in key
        }
        env.update(
            MUJOCO_GL=env.get("MUJOCO_GL", "egl"),
            PYOPENGL_PLATFORM=env.get("PYOPENGL_PLATFORM", "egl"),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMBA_CACHE_DIR=env.get(
                "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "llm_libero_numba")
            ),
        )
        self.worker_log = (self.output / "worker.log").open("a")
        self.process = subprocess.Popen(
            [str(self.libero_python), "-u", str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.worker_log,
            text=True,
            env=env,
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Load the scene's LIBERO task and init state, settle 10 steps, start a video."""
        meta = scene.metadata or {}
        task_id = meta.get("libero_task_id", self.task_id)
        if task_id is None:
            raise ConfigError(
                "this scene does not name a LIBERO task.\n"
                "fix: use --task libero_10 -T task_id=N, or pass -E task_id=N"
            )
        if self.task_id is not None and task_id != self.task_id:
            raise ConfigError(
                f"scene is LIBERO task {task_id} but the embodiment was built for task "
                f"{self.task_id}.\nfix: make -T task_id and -E task_id agree"
            )
        init_state = int(meta.get("init_state", self.init_state))
        env_seed = scene.init_seed if scene.init_seed is not None else ENV_SEED_BASE + init_state
        if self.process is None:
            self._start_worker()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.episode_dir = self.output / f"{stamp}_task{task_id:02d}_init{init_state:02d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        data = self._rpc(
            {
                "op": "reset",
                "task_id": int(task_id),
                "init_state": init_state,
                "env_seed": int(env_seed),
                "video": str(self.episode_dir / "rollout.mp4"),
            }
        )
        self.instruction = data["instruction"]
        return self._observation(data)

    def step(self, action: Action) -> StepResult:
        """Execute one 20 Hz absolute-pose action; terminate on LIBERO success."""
        data = self._rpc({"op": "step", "action": np.asarray(action.data).tolist()})
        success = bool(data["success"])
        return StepResult(
            self._observation(data),
            reward=float(success),
            terminated=success,
            termination_reason="success" if success else None,
            info={"success": success},
        )

    def close(self) -> None:
        """Stop the worker (flushing the last video); safe to call twice."""
        if self.process is None:
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write('{"op":"close"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.terminate()
            self.process.wait(timeout=10)
        finally:
            if self.worker_log is not None:
                self.worker_log.close()
            self.process = None
            self.worker_log = None

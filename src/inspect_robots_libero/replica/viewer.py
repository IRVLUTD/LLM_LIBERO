"""Live desktop copy of the simulator, isolated from offscreen EGL cameras.

Adapted from kochIRVL/sim/viewer.py; the source repository is not imported.
The child selects GLFW before importing MuJoCo, and mirrors scoring colors too.
"""
import multiprocessing as mp
import os
import threading
import time

import numpy as np


def _show_scene(scene, values, lock, ready, stop):
    os.environ["MUJOCO_GL"] = "glfw"
    import mujoco
    from mujoco import viewer

    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    existing_threads = set(threading.enumerate())
    with viewer.launch_passive(model, data) as handle:
        with handle.lock():
            handle.opt.sitegroup[:] = 0
            handle.cam.lookat[:] = [.20, 0, .10]
            handle.cam.distance = .70
            handle.cam.azimuth = 135
            handle.cam.elevation = -35
        ready.set()
        while handle.is_running() and not stop.is_set():
            with lock:
                snapshot = np.frombuffer(values, dtype=np.float64).copy()
            nq, nv, nu = model.nq, model.nv, model.nu
            with handle.lock():
                data.qpos[:] = snapshot[:nq]
                data.qvel[:] = snapshot[nq:nq+nv]
                data.ctrl[:] = snapshot[nq+nv:nq+nv+nu]
                model.geom_rgba[:] = snapshot[nq+nv+nu:-1].reshape(-1, 4)
                data.time = snapshot[-1]
                mujoco.mj_forward(model, data)
            handle.sync()
            time.sleep(1/30)
        stop.set()
    for thread in set(threading.enumerate())-existing_threads:
        thread.join(timeout=2)


class ViewerProcess:
    """View measured state only; all robot and object physics stays in the parent."""
    def __init__(self, simulation):
        self.sim = simulation
        context = mp.get_context("spawn")
        model = simulation.model
        self.values = context.RawArray("d", model.nq+model.nv+model.nu+model.ngeom*4+1)
        self.lock = context.Lock()
        self.ready, self.stop = context.Event(), context.Event()
        self.process = context.Process(target=_show_scene,
            args=(str(simulation.scene), self.values, self.lock, self.ready, self.stop),
            daemon=True, name="replica-live-viewer")
        self.sync()
        self.process.start()
        deadline = time.monotonic()+10
        while not self.ready.wait(.05):
            if not self.process.is_alive() or time.monotonic() >= deadline:
                self.close()
                raise RuntimeError("Live MuJoCo viewer could not open; check desktop DISPLAY or omit --viewer")
        if not self.process.is_alive():
            self.close()
            raise RuntimeError("Live MuJoCo viewer exited during startup")

    def sync(self):
        with self.lock:
            values = np.frombuffer(self.values, dtype=np.float64)
            values[:] = np.concatenate((self.sim.data.qpos, self.sim.data.qvel,
                self.sim.data.ctrl, self.sim.model.geom_rgba.ravel(), [self.sim.data.time]))

    def is_running(self):
        return self.process.is_alive() and not self.stop.is_set()

    def close(self):
        self.stop.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)

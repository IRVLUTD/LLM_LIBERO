"""Render and exercise real SO-101 physics with no API calls or keys."""
import argparse
import json
from pathlib import Path

import numpy as np
from inspect_robots.types import Action
from inspect_robots_libero.replica.embodiment import ReplicaEmbodiment, replica_task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--all-scenes", action="store_true", help="Default: one scene per task family")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--viewer", action="store_true", help="Open a passive MuJoCo viewer for one scene after checks; Escape/close to exit")
    p.add_argument("--scene-id", help="Check only this catalog scene")
    args = p.parse_args()
    entries = json.loads(args.catalog.read_text())["scenes"]
    if args.scene_id:
        entries = [entry for entry in entries if entry["id"] == args.scene_id]
        if not entries:
            p.error("Unknown scene ID")
    elif not args.all_scenes:
        seen, selected = set(), []
        for entry in entries:
            if entry["task_id"] not in seen:
                selected.append(entry)
                seen.add(entry["task_id"])
        entries = selected
    if args.viewer and len(entries) != 1:
        p.error("--viewer requires --scene-id")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for entry in entries:
        config = args.catalog.resolve().parent/entry["config"]
        env = ReplicaEmbodiment(output_dir=args.output/entry["id"], video=False, max_steps=100)
        try:
            obs = env.reset(replica_task(str(config)).scenes[0])
            start = obs.state["eef_state"].copy()
            for _ in range(10):
                env.step(Action(data=start))
            assert not env.last["success"], "Initial scene already scores as success"
            for fraction in np.linspace(0, 1, 25):
                target = start.copy()
                target[2] += .015*fraction
                env.step(Action(data=target))
            rise = float(env.sim.data.site_xpos[env.sim.ee_id][2]-start[2])
            assert rise > .008, f"Tool did not rise: {rise}"
            assert not any(w.number for w in env.sim.data.warning), "Unstable physics"
            results.append(dict(scene=entry["id"], passed=True, steps=env.sim.steps, tool_rise_m=rise))
            if args.viewer:
                # The headless renderer and GLFW viewer use separate processes.
                import subprocess
                import sys
                subprocess.run([sys.executable, "-m", "mujoco.viewer", "--mjcf", str(config.with_suffix('.xml'))], check=True,
                               env={**__import__('os').environ, "MUJOCO_GL": "glfw"})
        finally:
            env.close()
    report = dict(passed=all(r["passed"] for r in results), real_api_calls=0, scenes=results)
    (args.output/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(passed=report["passed"], scenes=len(results), real_api_calls=0)))


if __name__ == "__main__":
    main()

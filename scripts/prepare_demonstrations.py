"""Build demonstration storyboards (few-shot context) from the official LIBERO-10 HDF5.

Needs h5py and libero:

    python scripts/prepare_demonstrations.py --task-id 2 --num-demos 2 \
        --dataset-dir /path/to/LIBERO/datasets/libero_10

Picks ``--num-demos`` demos uniformly at random (``--seed``), samples 12 frames
from each (first and last included) and tiles them into one 1152x848 PNG per
demo: agentview left, wrist right, in the same orientation the policy sees
live. Writes ``manifest.json`` (with SHA-256 of every file) for
``run_budgeted_rollouts.py --demo-manifest``. No actions, proprioception or
labels are included, only RGB.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FRAMES = 12
TILE_W, TILE_H, LABEL_H = 384, 192, 20


def libero_10_task_name(task_id):
    """Native LIBERO-10 BDDL task name (the HDF5 file is ``<name>_demo.hdf5``)."""
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()["libero_10"]().get_task(task_id).name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="folder with the 10 LIBERO-10 *_demo.hdf5 files")
    parser.add_argument("--num-demos", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        help="default: outputs/demos/task<ID>_seed<SEED>_n<N>")
    args = parser.parse_args()

    name = libero_10_task_name(args.task_id)
    source = args.dataset_dir / f"{name}_demo.hdf5"
    out = args.output or ROOT / "outputs" / "demos" / (
        f"task{args.task_id:02d}_seed{args.seed}_n{args.num_demos}")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as f:
        demo_ids = sorted(int(k.split("_")[1]) for k in f["data"].keys())
        selected = random.Random(args.seed).sample(demo_ids, args.num_demos)
        manifest = {
            "task_id": args.task_id, "task_name": name, "seed": args.seed,
            "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "candidate_count": len(demo_ids), "sampling": "uniform random without replacement",
            "frames_per_demo": FRAMES,
            "frame_sampling": "uniform temporal sampling including first and last",
            "views": "each tile: agentview left (flipped upright), wrist right",
            "input_kind": "RGB storyboards only, no actions/proprioception/skill labels",
            "demos": [],
        }
        for demo_id in selected:
            obs = f["data"][f"demo_{demo_id}"]["obs"]
            agent = obs["agentview_rgb"][()]
            wrist = obs["eye_in_hand_rgb"][()]
            indices = np.linspace(0, len(agent) - 1, FRAMES).round().astype(int).tolist()
            canvas = Image.new("RGB", (3 * TILE_W, 4 * (TILE_H + LABEL_H)), "white")
            draw = ImageDraw.Draw(canvas)
            for n, idx in enumerate(indices):
                x, y = (n % 3) * TILE_W, (n // 3) * (TILE_H + LABEL_H)
                # Match the live observation: agentview flipped on both axes, wrist as-is.
                frame = np.concatenate([agent[idx][::-1, ::-1], wrist[idx]], axis=1)
                tile = Image.fromarray(frame).resize((TILE_W, TILE_H), Image.Resampling.LANCZOS)
                canvas.paste(tile, (x, y + LABEL_H))
                draw.text((x + 4, y + 4),
                          f"Demo {demo_id} | {n + 1}/{FRAMES} | frame {idx} | {idx / 20:.2f}s",
                          fill="black")
            dest = out / f"demo_{demo_id:03d}_storyboard.png"
            canvas.save(dest)
            manifest["demos"].append({
                "demo_id": demo_id, "num_frames": len(agent), "frame_indices": indices,
                "storyboard": str(dest),
                "storyboard_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("selected_demo_ids:", [d["demo_id"] for d in manifest["demos"]])
    print(out / "manifest.json")


if __name__ == "__main__":
    main()

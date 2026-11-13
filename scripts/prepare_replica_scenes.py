"""Create editable SO-101 scenes from a read-only VLA Replica tasks directory."""
import argparse
from pathlib import Path

from inspect_robots_libero.replica.scenes import compile_scene, prepare_scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compile", type=Path, metavar="SCENE_JSON", help="Rebuild one edited JSON scene into MJCF")
    args = parser.parse_args()
    if args.compile:
        print(compile_scene(args.compile))
    elif args.tasks_root and args.output:
        catalog = prepare_scenes(args.tasks_root, args.output)
        print(f"Created {len(catalog['scenes'])} scenes: {args.output/'catalog.json'}")
    else:
        parser.error("use --tasks-root and --output, or --compile SCENE_JSON")


if __name__ == "__main__":
    main()

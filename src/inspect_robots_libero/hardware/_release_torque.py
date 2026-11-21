"""Run torque-only cleanup in the selected LeRobot Python, without provider SDKs."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import types


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    protocol_out=sys.stdout
    sys.stdout=sys.stderr
    namespace=types.ModuleType("_so101_torque")
    namespace.__path__=[str(Path(__file__).resolve().parent)]
    sys.modules[namespace.__name__]=namespace
    report=importlib.import_module("_so101_torque.torque").release_torque(args.config,args.output)
    protocol_out.write(json.dumps(report)+"\n")
    protocol_out.flush()
    return 0 if report["torque_disabled"] is True else 1


if __name__=="__main__":
    raise SystemExit(main())

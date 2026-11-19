"""JSON-line hardware worker. Run with the existing LeRobot interpreter.

The private namespace avoids importing the parent package (and Inspect Robots)
into an environment that only needs the device-side dependencies.
"""
import importlib
import json
from pathlib import Path
import select
import signal
import sys
import traceback
import types


def main():
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    namespace = types.ModuleType("_so101_runtime")
    namespace.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[namespace.__name__] = namespace
    runtime_module = importlib.import_module("_so101_runtime.runtime")
    runtime = None
    def terminate(signum, frame):
        raise runtime_module.StopRequested("hardware worker termination requested")
    signal.signal(signal.SIGTERM, terminate)

    def reply(value):
        protocol_out.write(json.dumps(value, allow_nan=False)+"\n")
        protocol_out.flush()

    try:
        while True:
            if not select.select([sys.stdin], [], [], .1)[0]:
                if runtime is not None:
                    runtime.pump()
                continue
            line = sys.stdin.readline()
            if not line:
                break
            request = json.loads(line)
            operation = request["op"]
            if operation == "open":
                if runtime is not None:
                    raise ValueError("Worker is already open")
                runtime = runtime_module.HardwareRuntime(**request["kwargs"])
                reply(runtime.observation())
            elif operation == "close":
                break
            elif runtime is None:
                raise ValueError("Open the worker first")
            elif operation == "observe":
                reply(runtime.capture_observation())
            elif operation == "diagnostics":
                reply(runtime.diagnostics())
            elif operation == "setup":
                reply(runtime.setup())
            elif operation == "start":
                reply(runtime.start())
            elif operation == "plan":
                reply(runtime.plan(request["targets"]))
            elif operation == "step":
                reply(runtime.step(request["target"], request["plan_id"]))
            elif operation == "hold":
                runtime.hold()
                reply(runtime.capture_observation())
            else:
                raise ValueError(f"Unknown operation {operation}")
    except BaseException as exc:
        if runtime is not None:
            runtime.stopped = str(exc) or type(exc).__name__
        traceback.print_exc(file=sys.stderr)
        reply(dict(error=type(exc).__name__, detail=str(exc)))
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()

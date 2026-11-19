"""Run the standard JSON worker with the actual upstream SO-ARM runtime."""
import importlib
from pathlib import Path
import sys
import types


def main():
    """Select the upstream runtime without importing the evaluation policy stack."""
    namespace = types.ModuleType("_so101_runtime")
    namespace.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[namespace.__name__] = namespace
    base_runtime = importlib.import_module("_so101_runtime.runtime")
    upstream_runtime = importlib.import_module("_so101_runtime.upstream_runtime")
    base_runtime.HardwareRuntime = upstream_runtime.UpstreamRuntime
    importlib.import_module("_so101_runtime._worker").main()


if __name__ == "__main__":
    main()

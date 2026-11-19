# SO-101 hardware sources

The nominal kinematic URDF, `SOURCE.json`, and Apache-2.0 `LICENSE` are copied
from the user's `kochIRVL/real_robot/assets`. Upstream attribution and URDF hash
are retained in `SOURCE.json`. Only kinematic joints are parsed; robot visual
meshes are unnecessary for this backend.

`../controller.py` and `../kinematics.py` were copied from
`kochIRVL/real_robot` as the existing working hardware foundation. Their module
headers identify the origin. The default profile path now points to a local
example; initial position-hold writes use the instrumentable driver method.
New settings, runtime, worker and embodiment modules add whole-path preflight,
command logging, isolated-process execution, read-only modes, and the common
Inspect Robots LLM policy interface. No source module is imported from kochIRVL.

`mock_calibration.json` contains synthetic full-range motor travel and zero
offsets for offline tests only. It is not calibration for a real arm. The example
profile intentionally remains unverified. Live execution reads the explicitly
selected physical arm calibration and profile without modifying them.

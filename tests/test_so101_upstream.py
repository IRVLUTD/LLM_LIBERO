"""Use the actual pinned upstream class with test-double devices and local HTTP."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from inspect_robots.scene import Scene
from inspect_robots_libero.hardware.settings import ASSETS
from inspect_robots_libero.evaluation_report import report_rows

AVAILABLE = bool(importlib.util.find_spec("inspect_robots_so101") and importlib.util.find_spec("scipy") and importlib.util.find_spec("cv2"))
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(AVAILABLE, "Install requirements.so101-upstream.txt and hardware test dependencies")
class UpstreamBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        profile = json.loads((ASSETS/"profile.example.json").read_text())
        profile["urdf"] = str(ASSETS/"so101_new_calib.urdf")
        (self.root/"profile.json").write_text(json.dumps(profile))
        (self.root/"test_arm.json").write_text((ASSETS/"mock_calibration.json").read_text())
        config = json.loads((ASSETS/"mock_config.json").read_text())
        config.update(profile_file="profile.json", calibration_file="test_arm.json", follower_id="test_arm")
        self.config = self.root/"config.json"
        self.config.write_text(json.dumps(config))
        self.scene = self.root/"scene.json"
        self.scene.write_text(json.dumps(dict(schema_version=1, id="ID_task01_v01", task_id=1, subset="ID", variation=1,
            instruction="Put the bread on the red plate.", calibration={"status": "approximate"}, provenance={})))

    def runtime(self, control, **kwargs):
        from inspect_robots_libero.hardware.upstream_runtime import UpstreamRuntime
        runtime = UpstreamRuntime(self.config, self.root/(control+str(len(list(self.root.iterdir())))),
                                  control=control, video=False, **kwargs)
        self.addCleanup(runtime.close)
        runtime.setup()
        runtime.start()
        return runtime

    def test_source_pin_and_action_contracts(self):
        from inspect_robots_libero.hardware.upstream import SO101UpstreamEmbodiment
        from inspect_robots_libero.hardware.upstream_support import upstream_provenance
        self.assertEqual(upstream_provenance()["commit"], (ROOT/"upstream_so101_commit.txt").read_text().strip())
        with patch("subprocess.Popen", side_effect=AssertionError("Construction must not open devices")):
            joint = SO101UpstreamEmbodiment(config_path=self.config, control="joint")
            cart = SO101UpstreamEmbodiment(config_path=self.config, control="cartesian")
        self.assertEqual(joint.info.action_space.semantics.control_mode, "joint_pos")
        self.assertEqual(cart.info.action_space.semantics.control_mode, "eef_abs_pose")
        self.assertEqual(joint.info.action_space.semantics.dim_labels[0], "shoulder_pan")
        self.assertEqual(cart.info.action_space.semantics.dim_labels[0], "x")
        # Native joint control needs motor calibration, not a Cartesian-frame verification flag.
        native = SO101UpstreamEmbodiment(config_path=self.config, control="joint", mode="execute")
        self.assertIsNone(native.process)
        with self.assertRaisesRegex(ValueError, "verified"):
            SO101UpstreamEmbodiment(config_path=self.config, control="cartesian", mode="execute")

    def test_native_uses_upstream_step_and_never_solves_ik(self):
        from inspect_robots_so101.embodiment import SOArmEmbodiment
        from inspect_robots_libero.hardware.kinematics import MOTORS
        runtime = self.runtime("joint")
        self.assertIs(type(runtime.body), SOArmEmbodiment)
        start = np.array([runtime.driver.motors[n] for n in MOTORS])
        end = start.copy()
        end[0] += .5
        with patch.object(runtime.controller.kinematics, "solve", side_effect=AssertionError("Native path must not use IK")), \
             patch.object(runtime.body, "step", wraps=runtime.body.step) as step:
            planned = runtime.plan([start, end])
            self.assertTrue(planned["accepted"], planned)
            self.assertEqual(step.call_count, 0)
            runtime.step(start, planned["plan_id"])
            runtime.step(end, planned["plan_id"])
            self.assertEqual(step.call_count, 2)
        self.assertEqual(runtime.summary()["steps"], 2)
        self.assertEqual(runtime.upstream_steps, 2)

    def test_cartesian_uses_ik_and_executes_every_packet_through_upstream(self):
        runtime = self.runtime("cartesian")
        state = runtime.last_state
        start = np.array(state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
        end = start.copy()
        end[2] += .005
        path = np.linspace(start, end, 12)
        with patch.object(runtime.controller.kinematics, "solve", wraps=runtime.controller.kinematics.solve) as ik, \
             patch.object(runtime.body, "step", wraps=runtime.body.step) as step:
            plan = runtime.plan(path)
            self.assertTrue(plan["accepted"], plan)
            self.assertGreater(ik.call_count, 0)
            self.assertEqual(step.call_count, 0)
            for target in path:
                runtime.step(target, plan["plan_id"])
            self.assertEqual(step.call_count, runtime.controller.steps)
        self.assertGreater(runtime.upstream_steps, 0)
        self.assertAlmostEqual(runtime.last_state["eef_position_m"][2], end[2], delta=.005)

    def test_upstream_clamp_is_the_real_library_implementation(self):
        from inspect_robots.types import Action
        runtime = self.runtime("joint")
        target = runtime.body._cfg.high+100
        runtime.body.step(Action(data=target))  # Mock devices only; exercise upstream's independent backstop.
        np.testing.assert_allclose(list(runtime.driver.sent[-1].values()), runtime.body._cfg.high)

    def test_native_lag_correction_preserves_a_held_gripper_target(self):
        runtime = self.runtime("joint")
        initial = dict(runtime.driver.motors)
        initial["gripper.pos"] = 0.
        runtime.controller.last_motor_goal = initial
        runtime.controller.last_gripper_motor_goal = 0.
        runtime.controller.gripper_command = 1.
        runtime.driver.motors["shoulder_pan.pos"] -= 1.
        runtime.driver.motors["gripper.pos"] = 55.
        runtime.capture_observation()
        start = np.array(list(runtime.driver.motors.values()))
        end = start.copy()
        end[0] += .5
        end[5] = 0.
        path = np.linspace(start, end, 32)
        plan = runtime.plan(path)
        self.assertTrue(plan["accepted"], plan)
        for point in path:
            runtime.step(point, plan["plan_id"])
        self.assertTrue(all(command["gripper.pos"] == 0 for command in runtime.driver.sent))
        vectors = np.array([list(initial.values())]+[list(command.values()) for command in runtime.driver.sent])
        self.assertTrue(np.all(np.abs(np.diff(vectors, axis=0)) <= np.array([15]*5+[40])/20+1e-7))

    def test_dry_run_never_invokes_upstream_motion_or_writes(self):
        from inspect_robots_libero.hardware.controller import MockDriver
        import inspect_robots_libero.hardware.runtime as module
        class ReadOnlyDriver(MockDriver):
            def __init__(self, **kwargs):
                super().__init__()
            def send(self, action):
                raise AssertionError("Read-only driver was asked to move")
        with patch.object(module, "HardwareDriver", ReadOnlyDriver):
            runtime = self.runtime("joint", mode="dry-run")
            before = np.array(list(runtime.driver.motors.values()))
            target = before.copy()
            target[0] += .5
            plan = runtime.plan([target])
            self.assertTrue(plan["accepted"], plan)
            with patch.object(runtime.body, "step", side_effect=AssertionError("Dry-run cannot call motion")):
                runtime.step(target, plan["plan_id"])
                runtime.close()
            self.assertEqual(runtime.motor_writes, 0)
            self.assertEqual(runtime.upstream_steps, 0)
            np.testing.assert_array_equal(list(runtime.driver.motors.values()), before)

    def test_failed_upstream_observation_keeps_the_confirmed_motor_step_count(self):
        runtime = self.runtime("joint")
        target = np.array(list(runtime.driver.motors.values()))
        plan = runtime.plan([target])
        with patch.object(runtime.body, "_observe", side_effect=RuntimeError("Camera failed after command")):
            with self.assertRaisesRegex(RuntimeError, "Camera failed"):
                runtime.step(target, plan["plan_id"])
        self.assertEqual(runtime.summary()["steps"], 1)
        self.assertEqual(runtime.upstream_steps, 1)

    def test_real_policy_both_modes_all_efforts_and_spreadsheet(self):
        spec = importlib.util.spec_from_file_location("upstream_runner", ROOT/"scripts/run_replica_evaluation.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for control in ("joint", "cartesian"):
            out = self.root/(control+"_sweep")
            with patch.object(runner.httpx, "HTTPTransport", side_effect=AssertionError("Live network forbidden")):
                runner.main(["--benchmark", "so101-upstream", "--control", control, "--scene-json", str(self.scene),
                    "--hardware-config", str(self.config), "--hardware-mode", "mock", "--model", "openai/gpt-6-luna",
                    "--mock", "--no-video", "--output", str(out)])
            data = json.loads((out/"evaluation.json").read_text())
            rows, calls, summaries = report_rows(out, data)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["status"] == "finished" for row in rows), rows)
            self.assertTrue(all(row["upstream_steps"] == row["steps"] > 0 for row in rows))
            self.assertTrue(all(row["control_interface"] == control and row["real_api_calls"] == 0 for row in rows))
            self.assertTrue(all(row["simulated_seconds"] is None and row["human_success"] is None for row in rows))
            self.assertTrue(all(row["control_interface"] == control and row["successful_episodes"] is None for row in summaries))
            for row in rows:
                source = out/row["id"]/"api/ep00_call01_request.json"
                request = json.loads(source.read_text())
                names = [tool["name"] for tool in request["tools"]]
                self.assertIn("move_joints" if control == "joint" else "move_to", names)
            self.assertTrue((out/"evaluation.xlsx").is_file())


if __name__ == "__main__":
    unittest.main()

"""Offline hardware-pipeline verification. No test opens serial or real cameras."""
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from inspect_robots.scene import Scene
from inspect_robots.types import Action
from inspect_robots_libero.hardware.embodiment import SO101HardwareEmbodiment, hardware_task
from inspect_robots_libero.hardware.controller import HardwareDriver
from inspect_robots_libero.hardware.settings import ASSETS
from inspect_robots_libero.evaluation_report import report_rows

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = bool(importlib.util.find_spec("scipy") and importlib.util.find_spec("cv2"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(AVAILABLE, "Hardware offline tests need scipy and opencv")
class HardwareTests(unittest.TestCase):
    def test_servo_diagnostics_only_reads_and_records_partial_failures(self):
        reads = []
        class ReadOnlyBus:
            motors = {"shoulder_lift": SimpleNamespace(id=2, model="sts3215")}
            def read(self, register, motor, *, normalize):
                reads.append((register, motor, normalize))
                if register == "Present_Current":
                    raise OSError("register unavailable")
                return 16
            def write(self, *args, **kwargs):
                raise AssertionError("Diagnostics must not change the servo")
        driver = object.__new__(HardwareDriver)
        driver.robot = SimpleNamespace(bus=ReadOnlyBus())
        report = driver.read_diagnostics()
        self.assertEqual(report["writes_performed"], 0)
        self.assertTrue(reads)
        self.assertTrue(all(not normalized for _, _, normalized in reads))
        shoulder = report["motors"]["shoulder_lift"]
        self.assertEqual(shoulder["registers"]["P_Coefficient"], 16)
        self.assertEqual(shoulder["registers"]["Torque_Limit"], 16)
        self.assertEqual(set(shoulder["read_errors"]), {"Present_Current"})
        self.assertIn("OSError", shoulder["read_errors"]["Present_Current"])

    def test_diagnostics_worker_protocol_does_not_send_targets(self):
        out = self.root/"diagnostics"
        env = SO101HardwareEmbodiment(config_path=self.config, output_dir=out, video=False)
        self.addCleanup(env.close)
        env.reset(Scene(id="diagnostics", instruction="Offline diagnostic protocol test"))
        report = env._rpc({"op": "diagnostics"})
        self.assertEqual(report["source"], "mock_device")
        self.assertEqual(report["writes_performed"], 0)
        self.assertEqual(env.last["motor_writes"], 0)
        self.assertEqual(json.loads((out/"motor_diagnostics.json").read_text()), report)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profile = self.root/"profile.json"
        profile = json.loads((ASSETS/"profile.example.json").read_text())
        profile["urdf"] = str(ASSETS/"so101_new_calib.urdf")
        self.profile.write_text(json.dumps(profile))
        self.calibration = self.root/"test_arm.json"
        self.calibration.write_text((ASSETS/"mock_calibration.json").read_text())
        self.config = self.root/"config.json"
        config = json.loads((ASSETS/"mock_config.json").read_text())
        config.update(profile_file=str(self.profile), calibration_file=str(self.calibration), follower_id="test_arm")
        self.config.write_text(json.dumps(config))
        self.scene = self.root/"task.json"
        # Hardware needs no MJCF, object geometry or simulated success predicate.
        self.scene.write_text(json.dumps(dict(schema_version=1, id="ID_task01_v01", subset="ID", task_id=1,
            variation=1, instruction="Put the bread on the red plate.", calibration={"status": "approximate"}, provenance={})))

    def runtime(self, name="runtime", mode="mock", **kwargs):
        from inspect_robots_libero.hardware.runtime import HardwareRuntime
        runtime = HardwareRuntime(self.config, self.root/name, mode=mode, video=False, **kwargs)
        self.addCleanup(runtime.close)
        runtime.setup()
        runtime.start()
        return runtime

    def test_whole_path_ik_and_motor_speed_and_travel_limits(self):
        runtime = self.runtime()
        state = runtime.last_state
        start = np.array(state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
        end = start.copy()
        end[2] += .01
        before = np.array([runtime.driver.motors[n] for n in runtime.driver.motors])
        path = np.linspace(start, end, 15)
        plan = runtime.plan(path)
        self.assertTrue(plan["accepted"], plan)
        self.assertEqual(runtime.driver.sent, [], "Preflight must not send even its first motor waypoint")
        for target in path:
            runtime.step(target, plan["plan_id"])
        self.assertGreater(runtime.controller.steps, 0)
        np.testing.assert_allclose(runtime.last_state["eef_position_m"], end[:3], atol=.005)
        vectors = np.array([before]+[list(command.values()) for command in runtime.driver.sent])
        limit = np.array([15]*5+[40])/20
        self.assertTrue(np.all(np.abs(np.diff(vectors, axis=0)) <= limit+1e-7))
        self.assertTrue(np.all(vectors >= runtime.controller.motor_limits[:, 0]-.09))
        self.assertTrue(np.all(vectors <= runtime.controller.motor_limits[:, 1]+.09))

    def test_infeasible_tail_never_executes_feasible_prefix(self):
        runtime = self.runtime()
        state = runtime.last_state
        start = np.array(state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
        end = start.copy()
        end[0] = -.35
        result = runtime.plan([start, end])
        self.assertFalse(result["accepted"])
        self.assertEqual(runtime.driver.sent, [])
        result = runtime.step(start, 999)
        self.assertFalse(result["feedback"]["accepted"])
        self.assertEqual(runtime.driver.sent, [])

    def test_late_ik_failure_discards_previously_valid_waypoints(self):
        runtime = self.runtime()
        state = runtime.last_state
        start = np.array(state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
        first, second = start.copy(), start.copy()
        first[2] += .001
        second[2] += .002
        original = runtime.controller.kinematics.solve
        calls = 0
        def fail_second(position, orientation, seed, required, limits):
            nonlocal calls
            calls += 1
            return original(position, orientation, seed, required, limits) if calls == 1 else (seed, 1., 1.)
        with patch.object(runtime.controller.kinematics, "solve", side_effect=fail_second):
            result = runtime.plan([first, second])
        self.assertEqual(calls, 2)
        self.assertFalse(result["accepted"])
        self.assertEqual(runtime.driver.sent, [])
        self.assertFalse(runtime.approved)

    def test_tracking_failure_holds_and_never_sends_next_path_target(self):
        runtime = self.runtime()
        state = runtime.last_state
        start = np.array(state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
        end = start.copy()
        end[2] += .003
        path = np.linspace(start, end, 8)
        plan = runtime.plan(path)
        self.assertTrue(plan["accepted"])
        runtime.step(path[0], plan["plan_id"])
        sent_before = len(runtime.driver.sent)
        runtime.driver.motors["shoulder_pan.pos"] += 10
        result = runtime.step(path[1], plan["plan_id"])
        self.assertIn("tracking", result["stopped"].lower())
        self.assertFalse(runtime.approved)
        # The sole new write is a measured-position hold, not the next waypoint.
        self.assertEqual(len(runtime.driver.sent), sent_before+1)
        self.assertEqual(runtime.controller.steps, 1)
        self.assertEqual(runtime.hold_writes, 1)

    def test_stale_and_drifted_observations_block_before_writes(self):
        runtime = self.runtime()
        state = runtime.last_state
        target = np.array(state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
        runtime.controller.observed_at -= 61
        self.assertFalse(runtime.plan([target])["accepted"])
        runtime.capture_observation()
        runtime.driver.motors["shoulder_pan.pos"] += 4
        self.assertFalse(runtime.plan([target])["accepted"])
        self.assertEqual(runtime.driver.sent, [])

    def test_drift_after_preflight_blocks_first_command(self):
        runtime = self.runtime()
        state = runtime.last_state
        target = np.array(state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
        plan = runtime.plan([target])
        self.assertTrue(plan["accepted"])
        runtime.driver.motors["shoulder_pan.pos"] += 3
        result = runtime.step(target, plan["plan_id"])
        self.assertFalse(result["feedback"]["accepted"])
        self.assertEqual(runtime.driver.sent, [])

    def test_read_only_driver_never_writes_and_never_pretends_to_move(self):
        from inspect_robots_libero.hardware.controller import MockDriver
        import inspect_robots_libero.hardware.runtime as module
        class ReadOnlyDriver(MockDriver):
            def __init__(self, **kwargs):
                super().__init__()
            def send(self, motors):
                raise AssertionError("dry-run attempted a motor write")
        with patch.object(module, "HardwareDriver", ReadOnlyDriver):
            runtime = self.runtime(mode="dry-run")
            before = dict(runtime.driver.motors)
            state = runtime.last_state
            start = np.array(state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
            end = start.copy()
            end[2] += .005
            plan = runtime.plan(np.linspace(start, end, 12))
            self.assertTrue(plan["accepted"], plan)
            for target in np.linspace(start, end, 12):
                runtime.step(target, plan["plan_id"])
            runtime.close()
            self.assertEqual(runtime.driver.motors, before)
            self.assertEqual(runtime.motor_writes, 0)
            self.assertEqual(runtime.controller.steps, 0)
            self.assertGreater(runtime.planned_steps, 0)

    def test_closed_grasp_command_is_preserved_when_encoder_stops_apart(self):
        runtime = self.runtime()
        runtime.controller.gripper_command = 1.
        runtime.controller.last_gripper_motor_goal = 0.
        runtime.driver.motors["gripper.pos"] = 55.
        runtime.capture_observation()
        state = runtime.last_state
        start = np.array(state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
        end = start.copy()
        end[2] += .003
        path = np.linspace(start, end, 8)
        plan = runtime.plan(path)
        self.assertTrue(plan["accepted"], plan)
        for target in path:
            runtime.step(target, plan["plan_id"])
        self.assertTrue(all(command["gripper.pos"] == 0 for command in runtime.driver.sent))
        runtime.driver.motors["gripper.pos"] = 55.
        runtime.hold()
        self.assertEqual(runtime.driver.sent[-1]["gripper.pos"], 0.)

    def test_new_chunk_obeys_command_speed_even_with_encoder_lag(self):
        runtime = self.runtime()
        state = runtime.last_state
        start = np.array(state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
        plan = runtime.plan([start])
        runtime.step(start, plan["plan_id"])
        previous = np.array(list(runtime.driver.sent[-1].values()))
        runtime.driver.motors["shoulder_pan.pos"] -= 1.
        runtime.capture_observation()
        state = runtime.last_state
        target = np.array(state["eef_position_m"]+[state["pitch_rad"], state["roll_rad"], state["gripper_command"]])
        before = len(runtime.driver.sent)
        plan = runtime.plan([target])
        self.assertTrue(plan["accepted"], plan)
        runtime.step(target, plan["plan_id"])
        commands = np.array([previous]+[list(row.values()) for row in runtime.driver.sent[before:]])
        self.assertTrue(np.all(np.abs(np.diff(commands, axis=0)) <= np.array([15]*5+[40])/20+1e-7))

    def test_unverified_execute_gate_fails_without_connecting(self):
        with patch("subprocess.Popen", side_effect=AssertionError("worker must not start")):
            with self.assertRaisesRegex(ValueError, "verified"):
                SO101HardwareEmbodiment(config_path=self.config, mode="execute")
            env = SO101HardwareEmbodiment(config_path=self.config, mode="execute", assume_calibration=True)
            self.assertIn("calibration explicitly assumed: True", env.docs)
        self.assertFalse(json.loads(self.profile.read_text())["kinematics_verified"])

    def test_lagged_pose_next_tool_path_keeps_preflighted_waypoints(self):
        from inspect_robots.approver import DeltaLimitApprover
        from inspect_robots.errors import SafetyAbort
        from inspect_robots_agent._tools import build_toolset
        from inspect_robots_agent._llm import ToolCall
        runtime = self.runtime()

        class DirectHardware(SO101HardwareEmbodiment):
            """Exercise real adapter/runtime logic without opening a worker or device."""
            def _rpc(self, request):
                if request["op"] == "plan":
                    return runtime.plan(request["targets"])
                result = runtime.step(request["target"], request["plan_id"])
                self.last = {k: v for k, v in result.items() if k not in ("state", "images_png")}
                self._last_state = result["state"]
                return result

        output = self.root/"direct"
        output.mkdir()
        env = DirectHardware(config_path=self.config, output_dir=output, video=False)
        env.instruction = "Offline tracking-lag regression"
        observation = env._observation(runtime.capture_observation())
        tools = build_toolset(env.info.action_space, env.info.observation_space,
                              env.info.control_hz, pre_check=env.preflight)
        approver, store = env.make_approver(), {}

        def move(obs, name):
            tool = ToolCall(id=name, name="move_to", arguments=json.dumps(dict(
                targets={"z": float(obs.state["eef_state"][2]+.003)}, note="Offline small lift")))
            result = tools.execute(tool, obs)
            self.assertIsNone(result.error)
            return result.chunk.actions

        first = move(observation, "first")
        for action in first:
            observation = env.step(approver.review(action, store)).observation
        first_steps = runtime.controller.steps
        # The next observation can differ from the previous requested endpoint
        # on a real motor. Instant-following mocks previously hid this defect.
        runtime.driver.motors["shoulder_lift.pos"] += 1.
        observation = env._observation(runtime.capture_observation())
        second = move(observation, "second")
        legacy = DeltaLimitApprover(env.info.action_space).review(second[0], dict(store))
        self.assertFalse(np.array_equal(legacy.data, second[0].data), "Regression must reproduce the original clamp mismatch")
        for action in second:
            reviewed = approver.review(action, store)
            np.testing.assert_array_equal(reviewed.data, action.data)
            result = env.step(reviewed)
            self.assertFalse(result.terminated)
            observation = result.observation
        self.assertGreater(runtime.controller.steps, first_steps)
        self.assertTrue(observation.state["controller_feedback"][0])
        self.assertIsNone(env._command_rejection)
        # A measured-state refresh does not reset the limiter inside a chunk.
        env._action_reference = observation.state["eef_state"] + .1
        unsafe = second[-1].data.copy()
        unsafe[2] += .1
        with self.assertRaises(SafetyAbort):
            approver.review(Action(data=unsafe), store)

    def test_startup_reset_refreshes_gripper_command(self):
        target = dict(zip(("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"),
                          [0., -30., 40., 30., 0., 20.]))
        reset = self.root/"reset.json"
        reset.write_text(json.dumps(target))
        settings = json.loads(self.config.read_text())
        settings.update(reset_mode="fixed", reset_action_file=str(reset))
        self.config.write_text(json.dumps(settings))
        runtime = self.runtime()
        self.assertAlmostEqual(runtime.last_state["gripper_command"], .8)
        self.assertAlmostEqual(runtime.last_state["measured_gripper_closure"], .8)

    def test_protocol_mismatch_ends_episode_without_motor_commands(self):
        runtime = self.runtime()
        class DirectHardware(SO101HardwareEmbodiment):
            def _rpc(self, request):
                result = runtime.step(request["target"], request["plan_id"])
                self.last = {k: v for k, v in result.items() if k not in ("state", "images_png")}
                return result
        output = self.root/"mismatch"
        output.mkdir()
        env = DirectHardware(config_path=self.config, output_dir=output, video=False)
        env.instruction = "Offline protocol rejection"
        obs = env._observation(runtime.capture_observation())
        runtime.plan([obs.state["eef_state"]])
        env._plan_id = 999
        result = env.step(Action(data=obs.state["eef_state"]))
        self.assertTrue(result.terminated)
        self.assertIn("matching whole-path preflight", result.termination_reason)
        self.assertEqual(runtime.driver.sent, [])

    def test_worker_stop_is_checked_during_llm_idle_and_holds(self):
        out = self.root/"worker_stop"
        env = SO101HardwareEmbodiment(config_path=self.config, output_dir=out, video=False)
        self.addCleanup(env.close)
        env.reset(Scene(id="check", instruction="Infrastructure test"))
        (out/"STOP").touch()
        env.process.wait(timeout=4)
        env.close()
        summary = json.loads((out/"hardware_summary.json").read_text())
        self.assertIn("STOP", summary["stopped"])
        self.assertGreaterEqual(summary["hold_writes"], 1)
        self.assertEqual(summary["steps"], 0)

    def test_same_upstream_policy_all_efforts_no_live_requests(self):
        runner = load_script("run_replica_evaluation")
        out = self.root/"full"
        with patch.object(runner.httpx, "HTTPTransport", side_effect=AssertionError("network forbidden")):
            runner.main(["--benchmark", "hardware", "--scene-json", str(self.scene), "--hardware-config", str(self.config),
                "--hardware-mode", "mock", "--model", "openai/gpt-6-luna", "--mock", "--no-video", "--output", str(out)])
        data = json.loads((out/"evaluation.json").read_text())
        rows, calls, summaries = report_rows(out, data)
        self.assertEqual({row["effort"] for row in rows}, {"low", "medium", "high"})
        self.assertTrue(all(row["status"] == "finished" for row in rows), rows)
        self.assertTrue(all(row["llm_calls"] == 2 and row["real_api_calls"] == 0 for row in rows))
        self.assertTrue(all(row["steps"] > 0 and row["framework_steps"] > 0 for row in rows))
        self.assertTrue(all(row["human_success"] is None and row["environment_success"] is None for row in rows))
        self.assertTrue(all(row["simulated_seconds"] is None for row in rows))
        self.assertTrue(all(row["successful_episodes"] is None and row["scored_episodes"] == 0 for row in summaries))
        self.assertTrue((out/"evaluation.xlsx").is_file())

    def test_model_done_is_unverified_and_human_score_is_separate(self):
        runner = load_script("run_replica_evaluation")
        out = self.root/"done"
        def done_reply(model, wire):
            return httpx.MockTransport(lambda request: httpx.Response(200, json=dict(id="done", status="completed",
                output=[dict(type="function_call", id="fc", call_id="c", name="done", arguments=json.dumps(dict(summary="Finished", hindsight="none")))],
                usage=dict(input_tokens=10, output_tokens=10))))
        with patch.object(runner, "mock_transport", side_effect=done_reply):
            runner.main(["--benchmark", "hardware", "--scene-json", str(self.scene), "--hardware-config", str(self.config),
                "--model", "openai/gpt-6-luna", "--effort", "medium", "--mock", "--no-video", "--output", str(out)])
        data = json.loads((out/"evaluation.json").read_text())
        row = data["runs"][0]
        self.assertEqual(row["termination_reasons"], ["model_done_unverified"])
        self.assertIsNone(row["environment_success"])
        self.assertIsNone(row["human_success"])
        self.assertEqual(row["steps"], 0)
        script = load_script("so101_hardware")
        with self.assertRaises(SystemExit):
            script.main(["score", "--evaluation", str(out), "--run-id", row["id"], "--success", "yes"])
        # A synthetic report fixture stands in for a finished physical episode;
        # no hardware is used to test the label/export operation.
        data["runs"][0]["device_mode"] = "execute"
        (out/"evaluation.json").write_text(json.dumps(data))
        script.main(["score", "--evaluation", str(out), "--run-id", row["id"], "--success", "yes", "--note", "Human annotation fixture"])
        scored = json.loads((out/"evaluation.json").read_text())["runs"][0]
        self.assertTrue(scored["human_success"])
        self.assertIsNone(scored["environment_success"])
        self.assertEqual(scored["termination_reasons"], ["model_done_unverified"])


if __name__ == "__main__":
    unittest.main()

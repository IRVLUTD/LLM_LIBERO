"""Multi-scene session tests: fake devices and local HTTP only."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import numpy as np

from inspect_robots_libero.hardware.agent_session import create_session_directory, load_scene_queue
from inspect_robots_libero.hardware.controller import MockDriver
from inspect_robots_libero.hardware.runtime import HardwareRuntime
from inspect_robots_libero.hardware.settings import ASSETS
from inspect_robots_libero.evaluation_report import report_rows

ROOT=Path(__file__).resolve().parents[1]


def load_runner():
    spec=importlib.util.spec_from_file_location("agent_session_runner",ROOT/"scripts/run_vla_replica_agent.py")
    module=importlib.util.module_from_spec(spec)
    with patch.object(sys,"path",[str(ROOT/"scripts"),*sys.path]):
        spec.loader.exec_module(module)
    return module


class AgentSessionTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.runner=load_runner()
        profile=json.loads((ASSETS/"profile.example.json").read_text())
        profile["urdf"]=str(ASSETS/"so101_new_calib.urdf")
        (self.root/"profile.json").write_text(json.dumps(profile))
        (self.root/"test_arm.json").write_text((ASSETS/"mock_calibration.json").read_text())
        self.initial=dict(MockDriver().motors)
        self.initial["shoulder_pan.pos"]=2.
        (self.root/"initial.json").write_text(json.dumps(self.initial))
        config=json.loads((ASSETS/"mock_config.json").read_text())
        config.update(profile_file="profile.json",calibration_file="test_arm.json",follower_id="test_arm",
                      reset_mode="none",reset_action_file="initial.json")
        self.config=self.root/"config.json"
        self.config.write_text(json.dumps(config))
        self.catalog=self.root/"catalog.json"
        self.queue=self.root/"queue.json"
        self.reference=self.root/"reference.png"
        Image.new("RGB",(64,64),(200,20,20)).save(self.reference)
        self.make_catalog([(2,[1])])

    def make_catalog(self,tasks):
        entries=[]
        for task,scenes in tasks:
            for variation in scenes:
                ident=f"ID_task{task:02d}_v{variation:02d}"
                path=self.root/f"{ident}.json"
                path.write_text(json.dumps(dict(schema_version=1,id=ident,task_id=task,subset="ID",variation=variation,
                    instruction=f"Perform task {task} scene {variation}.",seed=1,calibration={"status":"approximate"},
                    objects=[],provenance={"reference_image":str(self.reference)})))
                entries.append(dict(config=path.name))
        self.catalog.write_text(json.dumps(dict(scenes=entries)))
        self.queue.write_text(json.dumps(dict(schema_version=1,subset="ID",tasks=[dict(task_id=t,scenes=s) for t,s in tasks])))

    def arguments(self):
        return ["--catalog",str(self.catalog),"--queue",str(self.queue),"--hardware-config",str(self.config),
                "--output-root",str(self.root/"outputs"),"--model","openai/gpt-6.1-sol","--effort","medium","--no-video"]

    def test_exact_default_queue_and_scene_order(self):
        expected=[(2,[1,3,4]),(4,[1,2,5]),(6,[1,2,4]),(9,[1,3,4]),(10,[1,3,4])]
        self.make_catalog(expected)
        scenes=load_scene_queue(self.catalog,ROOT/"configs/vla_replica_agent_session.json")
        self.assertEqual([(s["task_id"],s["variation"]) for s in scenes],[(t,v) for t,vs in expected for v in vs])
        self.assertEqual([s["scene_order"] for s in scenes],list(range(1,16)))

    def test_plan_only_does_not_open_devices_and_freezes_inputs(self):
        before=self.config.read_bytes()
        with patch("subprocess.Popen",side_effect=AssertionError("Planning must not start devices")):
            output=self.runner.main(self.arguments())
        data=json.loads((output/"evaluation.json").read_text())
        self.assertEqual(data["status"],"planned")
        self.assertEqual(data["runs"][0]["status"],"pending")
        self.assertEqual(self.config.read_bytes(),before)
        operator=json.loads((output/"hardware_operator.json").read_text())
        episode=json.loads((output/"hardware_episode.json").read_text())
        self.assertTrue(operator["reference_overlay"])
        self.assertFalse(episode["reference_overlay"])
        self.assertEqual(episode["reset_mode"],"none")
        frozen=json.loads((output/"hardware_profile.json").read_text())
        self.assertEqual(Path(frozen["urdf"]),output/"robot.urdf")
        self.assertTrue((output/"evaluation.xlsx").exists())

    def test_timestamp_folder_contains_agent_and_does_not_overwrite(self):
        one=create_session_directory(self.root,"openai/gpt-6.1-sol","medium")
        two=create_session_directory(self.root,"openai/gpt-6.1-sol","medium")
        self.assertNotEqual(one,two)
        self.assertIn("openai_gpt-6.1-sol_medium",one.name)

    def test_operator_reset_is_explicit_and_untimed_and_separate_from_agent_steps(self):
        runtime=HardwareRuntime(self.config,self.root/"operator",mode="mock",video=False)
        self.addCleanup(runtime.close)
        self.assertEqual(runtime.driver.sent,[])
        with self.assertRaisesRegex(ValueError,"operator"):
            runtime.operator_reset()
        runtime.operator_view(str(self.reference),"SETUP: test")
        self.assertEqual(runtime.controller.deadline,float("inf"))
        self.assertEqual(runtime.driver.sent,[])
        reply=runtime.operator_reset()
        self.assertTrue(reply["feedback"]["accepted"])
        self.assertGreater(reply["reset_control_steps"],0)
        self.assertEqual(reply["steps"],0)
        self.assertEqual(runtime.driver.motors,self.initial)
        runtime.start()
        self.assertFalse(runtime.operator_mode)
        with self.assertRaisesRegex(ValueError,"operator"):
            runtime.operator_reset()

    def test_operator_reference_overlay_does_not_modify_policy_images(self):
        runtime=HardwareRuntime(self.config,self.root/"overlay",mode="mock",video=False)
        self.addCleanup(runtime.close)
        runtime.preview_enabled=True
        original={k:v.copy() for k,v in runtime.last_images.items()}
        with patch("cv2.imshow"),patch("cv2.waitKey",return_value=-1),patch("cv2.getWindowProperty",return_value=1):
            runtime.operator_view(str(self.reference),"SETUP: test")
        for name in original:
            np.testing.assert_array_equal(runtime.last_images[name],original[name])
        runtime.preview_open=False

    def test_mock_motor_pose_survives_worker_reconnection(self):
        path=self.root/"mock_state.json"
        first=MockDriver(state_file=path)
        first.send(self.initial)
        second=MockDriver(state_file=path)
        self.assertEqual(second.read_motors(),self.initial)

    def test_retry_preserves_attempts_costs_and_fresh_context(self):
        script=self.root/"answers.json"
        script.write_text(json.dumps(["r","","f","r","t","","s","r","n"]))
        caps=[]
        original=self.runner.single.main
        def episode(argv):
            caps.append(float(argv[argv.index("--total-budget")+1]))
            self.assertIn("--keep-torque",argv)
            return original(argv)
        release=self.runner.single.prompt_hardware_torque_release
        with patch.object(self.runner.single.httpx,"HTTPTransport",side_effect=AssertionError("Network forbidden")), \
             patch.object(self.runner.single,"main",side_effect=episode), \
             patch.object(self.runner.single,"prompt_hardware_torque_release",wraps=release) as end_prompt:
            output=self.runner.main(self.arguments()+["--mock","--operator-script",str(script)])
        self.assertEqual(end_prompt.call_count,1)
        data=json.loads((output/"evaluation.json").read_text())
        self.assertEqual(data["status"],"completed")
        self.assertEqual([r["attempt"] for r in data["runs"]],[1,2])
        self.assertEqual([r["operator_verdict"] for r in data["runs"]],["failure","success"])
        self.assertTrue(all(r["human_success"] is None for r in data["runs"]))
        self.assertAlmostEqual(data["budget_charge_usd"],sum(r["budget_charge_usd"] for r in data["runs"]))
        self.assertAlmostEqual(caps[1],caps[0]-data["runs"][0]["budget_charge_usd"])
        episodes,calls,_=report_rows(output,data)
        self.assertEqual(len(calls),4)
        self.assertTrue(all(r["real_api_calls"]==0 and r["llm_calls"]==2 for r in episodes))
        for row in data["runs"]:
            request=json.loads((output/row["run_folder"]/"api/ep00_call01_request.json").read_text())
            self.assertFalse(any(item.get("type")=="function_call" for item in request["input"]))
            trajectory=json.loads((output/row["run_folder"]/"trajectory.json").read_text())
            self.assertAlmostEqual(trajectory[0]["motor_positions"][0],2.)
            self.assertTrue((output/row["run_folder"]/"hardware/motor_events.jsonl").exists())
            self.assertEqual(row["reset_control_steps"],0)

    def test_hardware_fault_pauses_instead_of_starting_the_next_scene(self):
        self.make_catalog([(2,[1,3])])
        script=self.root/"fault.json"
        script.write_text(json.dumps(["r","","f","q"]))
        original=self.runner.single.main
        def fault(argv):
            original(argv)
            directory=Path(argv[argv.index("--output")+1])
            path=directory/"evaluation.json"
            data=json.loads(path.read_text())
            data["runs"][0].update(status="hardware_stopped",hardware_stop_detail="Injected tracking fault")
            path.write_text(json.dumps(data))
        with patch.object(self.runner.single,"main",side_effect=fault) as episodes:
            output=self.runner.main(self.arguments()+["--mock","--operator-script",str(script)])
        data=json.loads((output/"evaluation.json").read_text())
        self.assertEqual(episodes.call_count,1)
        self.assertEqual(data["status"],"stopped_operator")
        self.assertEqual(data["runs"][1]["status"],"not_run")
        events=[json.loads(s) for s in (output/"session_events.jsonl").read_text().splitlines()]
        self.assertTrue(any(e["event"]=="hardware_fault_pause" for e in events))

    def test_unknown_usage_blocks_more_api_attempts(self):
        self.make_catalog([(2,[1,3])])
        script=self.root/"unknown.json"
        script.write_text(json.dumps(["r","","f",""]))
        original=self.runner.single.main
        def unknown(argv):
            original(argv)
            directory=Path(argv[argv.index("--output")+1])
            path=directory/"evaluation.json"
            data=json.loads(path.read_text())
            data["runs"][0].update(status="usage_unknown",usage_unknown=True)
            path.write_text(json.dumps(data))
        with patch.object(self.runner.single,"main",side_effect=unknown) as episodes:
            output=self.runner.main(self.arguments()+["--mock","--operator-script",str(script)])
        data=json.loads((output/"evaluation.json").read_text())
        self.assertEqual(episodes.call_count,1)
        self.assertEqual(data["status"],"stopped_usage_unknown")
        self.assertEqual(data["runs"][1]["status"],"not_run")

    def test_reset_key_is_required_before_start_and_before_next_trial(self):
        script=self.root/"reset.json"
        script.write_text(json.dumps(["","r","","s","n","r","n"]))
        output=self.runner.main(self.arguments()+["--mock","--operator-script",str(script)])
        data=json.loads((output/"evaluation.json").read_text())
        self.assertEqual(data["status"],"completed")
        events=[json.loads(s) for s in (output/"session_events.jsonl").read_text().splitlines()]
        self.assertLess(next(i for i,e in enumerate(events) if e["event"]=="reset_finished"),
                        next(i for i,e in enumerate(events) if e["event"]=="attempt_started"))
        self.assertEqual(sum(e["event"]=="attempt_started" for e in events),1)

    def test_skip_does_not_create_a_fake_failure_or_api_calls(self):
        script=self.root/"skip.json"
        script.write_text(json.dumps(["k"]))
        output=self.runner.main(self.arguments()+["--mock","--operator-script",str(script)])
        data=json.loads((output/"evaluation.json").read_text())
        self.assertEqual(data["status"],"completed")
        self.assertEqual(data["runs"][0]["status"],"not_run")
        self.assertIsNone(data["runs"][0]["human_success"])
        self.assertEqual(data["budget_charge_usd"],0)

    @unittest.skipUnless(importlib.util.find_spec("imageio_ffmpeg"),"Video integration needs imageio-ffmpeg")
    def test_default_video_artifacts_are_saved_for_agent_and_operator_stages(self):
        script=self.root/"video.json"
        script.write_text(json.dumps(["r","","s","r","n"]))
        arguments=self.arguments()
        arguments.remove("--no-video")
        output=self.runner.main(arguments+["--mock","--operator-script",str(script)])
        data=json.loads((output/"evaluation.json").read_text())
        self.assertEqual(data["status"],"completed")
        video=output/data["runs"][0]["run_folder"]/"hardware/rollout.mp4"
        self.assertGreater(video.stat().st_size,1000)
        operator_videos=list((output/"operator").rglob("rollout.mp4"))
        self.assertGreaterEqual(len(operator_videos),2)
        self.assertTrue(all(p.stat().st_size>1000 for p in operator_videos))

    def test_mock_operator_script_cannot_drive_real_hardware(self):
        script=self.root/"unsafe.json"
        script.write_text(json.dumps(["r",""]))
        with patch("subprocess.Popen",side_effect=AssertionError("Must reject before devices")):
            with self.assertRaises(SystemExit):
                self.runner.main(self.arguments()+["--mock","--hardware-mode","execute","--operator-script",str(script)])


if __name__=="__main__":
    unittest.main()

"""Post-episode handoff tests: fake buses only, never real torque writes."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from inspect_robots_libero.hardware import torque
from inspect_robots_libero.hardware.embodiment import SO101HardwareEmbodiment
from inspect_robots_libero.hardware.settings import ASSETS, MOTOR_KEYS

ROOT=Path(__file__).resolve().parents[1]


class FakeBus:
    def __init__(self):
        self.motors={name.removesuffix(".pos"):object() for name in MOTOR_KEYS}
        self.is_connected=False
        self.is_calibrated=True
        self.calls=[]
        self.failed_motor=None

    def connect(self):
        self.calls.append("connect")
        self.is_connected=True

    def disable_torque(self):
        self.calls.append("disable_torque")

    def read(self,register,motor):
        self.calls.append(("read",register,motor))
        if register!="Torque_Enable":
            raise AssertionError("Unexpected servo register")
        return 1 if motor==self.failed_motor else 0

    def disconnect(self,*,disable_torque):
        self.calls.append(("disconnect",disable_torque))
        self.is_connected=False


class TorqueReleaseTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        (self.root/"test_arm.json").write_text((ASSETS/"mock_calibration.json").read_text())
        settings=json.loads((ASSETS/"mock_config.json").read_text())
        settings.update(profile_file=str(ASSETS/"profile.example.json"),
            calibration_file=str(self.root/"test_arm.json"),follower_id="test_arm",port=str(self.root/"fake_serial"))
        self.config=self.root/"config.json"
        self.config.write_text(json.dumps(settings))
        spec=importlib.util.spec_from_file_location("torque_runner",ROOT/"scripts/run_replica_evaluation.py")
        self.runner=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)
        self.bus=FakeBus()

    def environment(self):
        out=self.root/"episode"
        out.mkdir(exist_ok=True)
        (out/"worker.log").touch()
        return SimpleNamespace(mode="execute",output=out,
            release_torque_after_close=Mock(return_value=dict(status="disabled",torque_disabled=True)))

    def test_release_only_disables_and_checks_all_six_motors(self):
        with patch.object(torque,"_make_robot",return_value=SimpleNamespace(bus=self.bus)):
            report=torque.release_torque(self.config,self.root/"release.json")
        self.assertTrue(report["torque_disabled"])
        self.assertEqual(report["motor_position_writes"],0)
        self.assertEqual(report["torque_on_writes"],0)
        self.assertEqual(report["torque_disable_requests"],1)
        self.assertEqual(len(report["torque_enable_readback"]),6)
        self.assertEqual(self.bus.calls[0:2],["connect","disable_torque"])
        self.assertEqual(self.bus.calls[-1],("disconnect",False))
        self.assertEqual(json.loads((self.root/"release.json").read_text()),report)

    def test_calibration_mismatch_never_writes_torque(self):
        self.bus.is_calibrated=False
        with patch.object(torque,"_make_robot",return_value=SimpleNamespace(bus=self.bus)):
            report=torque.release_torque(self.config,self.root/"release.json")
        self.assertEqual(report["status"],"failed")
        self.assertNotIn("disable_torque",self.bus.calls)
        self.assertIsNone(report["torque_disabled"])

    def test_partial_release_is_not_reported_as_success(self):
        self.bus.failed_motor="wrist_flex"
        with patch.object(torque,"_make_robot",return_value=SimpleNamespace(bus=self.bus)):
            report=torque.release_torque(self.config,self.root/"release.json")
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["torque_enable_readback"]["wrist_flex"],1)
        self.assertIsNone(report["torque_disabled"])

    def test_enter_requests_release_and_records_confirmation(self):
        env=self.environment()
        row={}
        with patch.object(self.runner.sys,"stdin",SimpleNamespace(isatty=lambda:True)),patch("builtins.input",return_value=""):
            self.runner.prompt_hardware_torque_release(env,row)
        env.release_torque_after_close.assert_called_once_with()
        self.assertTrue(row["torque_disabled"])

    def test_keep_eof_and_nonterminal_never_release(self):
        env=self.environment()
        cases=[("keep",None,True),(None,EOFError(),True),(None,None,False)]
        for answer,error,terminal in cases:
            row={}
            with patch.object(self.runner.sys,"stdin",SimpleNamespace(isatty=lambda:terminal)),patch("builtins.input",return_value=answer,side_effect=error):
                self.runner.prompt_hardware_torque_release(env,row)
            self.assertTrue(row["torque_release_status"].startswith("kept"))
            env.release_torque_after_close.assert_not_called()

    def test_mock_dryrun_and_keep_option_do_not_prompt(self):
        env=self.environment()
        with patch("builtins.input",side_effect=AssertionError("Unexpected prompt")):
            self.runner.prompt_hardware_torque_release(env,{},keep_torque=True)
            for mode in ("mock","dry-run"):
                env.mode=mode
                self.runner.prompt_hardware_torque_release(env,{})
        env.release_torque_after_close.assert_not_called()

    def test_next_trial_waits_after_release_and_can_be_cancelled(self):
        for answer,want in (("",True),("q",False)):
            with patch("builtins.input",return_value=answer):
                self.assertEqual(self.runner.wait_for_next_hardware_episode(),want)
        with patch("builtins.input",side_effect=EOFError()):
            self.assertFalse(self.runner.wait_for_next_hardware_episode())

    def test_public_release_rejects_dryrun_without_spawning(self):
        env=SO101HardwareEmbodiment(config_path=self.config,mode="dry-run",output_dir=self.root/"dry")
        with patch("subprocess.run",side_effect=AssertionError("Must not start a helper")):
            with self.assertRaisesRegex(ValueError,"physical execute"):
                env.release_torque_after_close()

    def test_public_release_uses_isolated_helper_after_close(self):
        out=self.root/"episode"
        out.mkdir()
        env=SO101HardwareEmbodiment(config_path=self.config,mode="execute",assume_calibration=True,output_dir=out)
        (out/"hardware_summary.json").write_text(json.dumps(dict(steps=2)))
        def run(command,**kwargs):
            self.assertTrue(command[2].endswith("_release_torque.py"))
            self.assertNotIn("TEST_API_KEY",kwargs["env"])
            config=json.loads((out/"torque_release_config.json").read_text())
            self.assertEqual(config["port"],env.settings["port"])
            (out/"torque_release.json").write_text(json.dumps(dict(status="disabled",torque_disabled=True)))
            return SimpleNamespace(returncode=0)
        with patch.dict("os.environ",{"TEST_API_KEY":"never-forward-this"}),patch("subprocess.run",side_effect=run):
            result=env.release_torque_after_close()
        self.assertTrue(result["torque_disabled"])
        summary=json.loads((out/"hardware_summary.json").read_text())
        self.assertEqual(summary["steps"],2)
        self.assertTrue(summary["torque_disabled"])

    def test_runner_saves_and_closes_before_prompt_on_done_and_error(self):
        scene=self.root/"scene.json"
        scene.write_text(json.dumps(dict(schema_version=1,id="ID_task01_v01",task_id=1,subset="ID",variation=1,
            instruction="Put the bread on the plate.",calibration={"status":"approximate"},provenance={})))
        for fail in (False,True):
            with self.subTest(evaluation_error=fail):
                events=[]
                out=self.root/f"integration-{fail}"
                def factory(**kwargs):
                    requested=kwargs.pop("mode")
                    kwargs.pop("assume_calibration",None)
                    env=SO101HardwareEmbodiment(mode="mock",**kwargs)
                    env.mode=requested
                    if "output_dir" in kwargs:
                        env.output.mkdir(parents=True)
                        (env.output/"worker.log").touch()
                    env.close=lambda:events.append("closed")
                    def release():
                        events.append("released")
                        return dict(status="disabled",torque_disabled=True)
                    env.release_torque_after_close=release
                    return env
                def evaluate(task,policy,env,**kwargs):
                    env.last=dict(steps=0,device_mode="execute")
                    if fail:
                        raise RuntimeError("Synthetic evaluation fault; no devices were opened")
                    return (SimpleNamespace(status="success",samples=[SimpleNamespace(termination_reasons=["model_done_unverified"])]),)
                saved_wall=[]
                def answer(prompt):
                    self.assertIn("closed",events)
                    saved=json.loads((out/"evaluation.json").read_text())
                    self.assertIn(saved["runs"][0]["status"],("finished","error"))
                    saved_wall.append(saved["runs"][0]["wall_seconds"])
                    events.append("prompted")
                    return ""
                import httpx
                with patch.object(self.runner,"SO101HardwareEmbodiment",side_effect=factory), \
                     patch.object(self.runner,"evaluate",side_effect=evaluate), \
                     patch.object(self.runner,"agent_policy",return_value=SimpleNamespace(_client=SimpleNamespace(close=lambda:None))), \
                     patch.object(self.runner.httpx,"HTTPTransport",return_value=httpx.MockTransport(lambda request: (_ for _ in ()).throw(AssertionError("Network forbidden")))), \
                     patch.dict("os.environ",{"OPENAI_API_KEY":"offline-fixture"}), \
                     patch.object(self.runner.sys,"stdin",SimpleNamespace(isatty=lambda:True)), \
                     patch("builtins.input",side_effect=answer):
                    self.runner.main(["--benchmark","hardware","--scene-json",str(scene),"--hardware-config",str(self.config),
                        "--hardware-mode","execute","--assume-calibration","--model","openai/gpt-6.1-sol","--effort","medium",
                        "--execute-paid-rollouts","--per-run-budget","1","--total-budget","1","--no-video","--output",str(out)])
                row=json.loads((out/"evaluation.json").read_text())["runs"][0]
                self.assertTrue(row["torque_disabled"])
                self.assertEqual(row["torque_release_status"],"disabled")
                self.assertEqual(row["wall_seconds"],saved_wall[0])
                self.assertLess(events.index("closed"),events.index("prompted"))
                self.assertLess(events.index("prompted"),events.index("released"))


if __name__=="__main__":
    unittest.main()

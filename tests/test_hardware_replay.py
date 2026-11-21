"""Replay transfer tests use synthetic recordings and mock devices only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inspect_robots_libero.hardware.controller import MockDriver
from inspect_robots_libero.hardware.kinematics import Kinematics, MOTORS, load_profile
from inspect_robots_libero.hardware.replay import ReplayRejected, identity_frame, load_recording, map_targets, plan_replay, retarget_pitch_height
from inspect_robots_libero.hardware.runtime import HardwareRuntime
from inspect_robots_libero.hardware.settings import ASSETS


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.profile=load_profile(ASSETS/"profile.example.json")
        self.calibration=json.loads((ASSETS/"mock_calibration.json").read_text())
        self.start=MockDriver().read_motors()
        kin=Kinematics(self.profile)
        pose=kin.forward(kin.motor_to_model(self.start))
        closure=(self.start["gripper.pos"]-100)/-100
        self.pose=np.r_[pose[:3,3],kin.orientation(pose),closure]
        self.targets=np.tile(self.pose,(3,1))
        self.targets[:,2]+=np.linspace(0,.004,3)
        (self.root/"profile.json").write_text(json.dumps(self.profile))
        (self.root/"test_arm.json").write_text(json.dumps(self.calibration))
        (self.root/"start.json").write_text(json.dumps(self.start))
        config=json.loads((ASSETS/"mock_config.json").read_text())
        config.update(profile_file="profile.json",calibration_file="test_arm.json",follower_id="test_arm",
                      reset_mode="none",reset_action_file="start.json")
        self.config=self.root/"hardware.json"
        self.config.write_text(json.dumps(config))

    def recording(self):
        episode=self.root/"episode"
        (episode/"sim").mkdir(parents=True)
        (episode/"sim/scene.json").write_text(json.dumps(dict(id="ID_task02_v01",instruction="Put the bowl on the coaster.",objects=[])))
        (episode/"sim/score.json").write_text(json.dumps(dict(success=False)))
        actions=episode/"inspect/actions/run"
        actions.mkdir(parents=True)
        rows=[dict(kind="header",labels=["x","y","z","pitch","roll","gripper"])]
        rows.extend(dict(t=i,action=p.tolist()) for i,p in enumerate(self.targets))
        (actions/"actions.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        (episode/"trajectory.json").write_text(json.dumps([dict(step=i+1,eef=p.tolist()) for i,p in enumerate(self.targets)]))
        return episode

    def runtime(self):
        runtime=HardwareRuntime(self.config,self.root/"hardware",video=False)
        self.addCleanup(runtime.close)
        runtime.setup()
        runtime.start()
        return runtime

    def test_source_track_and_invalid_action_schema(self):
        episode=self.recording()
        commands,metadata=load_recording(episode)
        observed,_=load_recording(episode,"observed")
        np.testing.assert_allclose(commands,observed)
        self.assertFalse(metadata["source_success"])
        f=next((episode/"inspect/actions").rglob("*.jsonl"))
        f.write_text(f.read_text().replace('"pitch"','"joint_4"'))
        with self.assertRaisesRegex(ValueError,"Cartesian"):
            load_recording(episode)

    def test_frame_mapping_is_explicit_and_does_not_change_closure(self):
        frame=identity_frame()
        frame.update(yaw_deg=90,translation_m=[.1,.2,.3])
        mapped=map_targets(self.targets,frame)
        np.testing.assert_allclose(mapped[:,0],-self.targets[:,1]+.1)
        np.testing.assert_allclose(mapped[:,1],self.targets[:,0]+.2)
        np.testing.assert_allclose(mapped[:,2],self.targets[:,2]+.3)
        np.testing.assert_array_equal(mapped[:,3:],self.targets[:,3:])

    def test_resume_keeps_original_source_indices(self):
        episode=self.recording()
        targets,metadata=load_recording(episode,start_step=2,limit=1)
        np.testing.assert_array_equal(targets,self.targets[1:2])
        self.assertEqual(metadata["source_start_step"],2)
        self.assertEqual(metadata["source_end_step"],2)
        result=plan_replay(self.profile,self.calibration,targets,self.start,source_start_step=2)
        self.assertEqual(set(result["source_steps"]),{2})

    def test_resume_rejection_reports_global_source_step(self):
        targets=self.targets.copy()
        targets[-1,2]=.008
        with self.assertRaises(ReplayRejected) as caught:
            plan_replay(self.profile,self.calibration,targets,self.start,source_start_step=29)
        self.assertEqual(caught.exception.report["issues"][0]["source_step"],31)

    def test_retargeting_preserves_xy_roll_and_gripper_and_requires_strict_preflight(self):
        adjusted,report=retarget_pitch_height(self.profile,self.calibration,self.targets,self.start,source_start_step=23)
        np.testing.assert_array_equal(adjusted[:,[0,1,4,5]],self.targets[:,[0,1,4,5]])
        self.assertEqual(report["changes"][0]["source_step"],23)
        plan=plan_replay(self.profile,self.calibration,adjusted,self.start,source_start_step=23)
        self.assertTrue(plan["accepted"])

    def test_retargeting_raises_low_heights_without_lowering_the_floor(self):
        profile=json.loads(json.dumps(self.profile))
        profile["workspace_m"]["z"][0]=float(self.pose[2]+.001)
        original_floor=profile["workspace_m"]["z"][0]
        adjusted,report=retarget_pitch_height(profile,self.calibration,self.targets,self.start)
        self.assertTrue(np.all(adjusted[:,2]>=original_floor+profile["position_tolerance_m"]+.001-1e-9))
        self.assertEqual(profile["workspace_m"]["z"][0],original_floor)
        self.assertGreater(report["summary"]["height_adjusted_samples"],0)

    def test_retiming_respects_all_native_motor_speeds(self):
        targets=self.targets.copy()
        targets[-1,5]=.9
        result=plan_replay(self.profile,self.calibration,targets,self.start,time_scale=3)
        commands=np.array(result["motor_targets"])
        increments=np.diff(np.vstack([list(self.start.values()),commands]),axis=0)
        maximum=np.array([self.profile["joint_speed_deg_s"]]*5+[self.profile["gripper_speed_units_s"]])/self.profile["control_hz"]
        self.assertTrue(np.all(np.abs(increments)<=maximum+1e-8))
        self.assertGreaterEqual(result["summary"]["motor_steps"]-result["summary"]["approach_motor_steps"],6)
        self.assertAlmostEqual(commands[-1,5],10)

    def test_invalid_tail_rejects_whole_path_before_any_motor_write(self):
        runtime=self.runtime()
        targets=self.targets.copy()
        targets[-1,2]=.008
        result=runtime.prepare_replay(targets)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["issues"][0]["source_step"],3)
        self.assertIsNone(runtime.replay_plan)
        self.assertEqual(runtime.driver.sent,[])

    def test_late_ik_failure_does_not_leave_an_executable_prefix(self):
        runtime=self.runtime()
        original=Kinematics.solve
        calls=0
        def solve(kin,*args):
            nonlocal calls
            calls+=1
            return original(kin,*args) if calls==1 else (args[2],1.,1.)
        with patch.object(Kinematics,"solve",solve):
            result=runtime.prepare_replay(self.targets)
        self.assertFalse(result["accepted"])
        self.assertEqual(runtime.driver.sent,[])
        self.assertIsNone(runtime.replay_plan)

    def test_replay_uses_cached_plan_and_records_actual_tracking(self):
        runtime=self.runtime()
        plan=runtime.prepare_replay(self.targets)
        self.assertTrue(plan["accepted"],plan)
        self.assertEqual(runtime.driver.sent,[])
        result=runtime.run_replay(plan["plan_id"])
        self.assertEqual(result["status"],"replayed")
        self.assertEqual(result["real_api_calls"],0)
        self.assertEqual(result["agent_cost_usd"],0)
        self.assertEqual(result["peak_arm_tracking_error_deg"],0)
        self.assertEqual(len(runtime.driver.sent),result["completed_motor_steps"])
        self.assertTrue((runtime.output/"replay_tracking.csv").exists())
        self.assertFalse(runtime.run_replay(plan["plan_id"])["accepted"])

    def test_drift_invalidates_single_use_plan_before_sending(self):
        runtime=self.runtime()
        plan=runtime.prepare_replay(self.targets)
        runtime.driver.motors["shoulder_pan.pos"]+=3
        result=runtime.run_replay(plan["plan_id"])
        self.assertFalse(result["accepted"])
        self.assertIn("moved",result["reason"])
        self.assertEqual(runtime.driver.sent,[])
        self.assertFalse(runtime.run_replay(plan["plan_id"])["accepted"])

    def test_tracking_stop_blocks_remaining_replay(self):
        runtime=self.runtime()
        plan=runtime.prepare_replay(self.targets)
        original=runtime.driver.send
        def lag_first(target):
            result=original(target)
            if runtime.motor_writes==1:
                runtime.driver.motors["shoulder_pan.pos"]+=7
            return result
        runtime.driver.send=lag_first
        result=runtime.run_replay(plan["plan_id"])
        self.assertEqual(result["status"],"hardware_stopped")
        self.assertEqual(result["completed_motor_steps"],1)
        self.assertIn("tracking",result["stop_detail"].lower())
        self.assertLess(result["completed_motor_steps"],plan["motor_steps"])

    def test_prepare_cli_never_starts_a_device_process(self):
        episode=self.recording()
        script=Path(__file__).resolve().parents[1]/"scripts/replay_replica_trajectory.py"
        spec=importlib.util.spec_from_file_location("replay_cli",script)
        cli=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        out=self.root/"prepared"
        with patch("subprocess.Popen",side_effect=AssertionError("Offline preparation cannot open devices")):
            code=cli.main(["--source",str(episode),"--hardware-config",str(self.config),"--output",str(out)])
        self.assertEqual(code,0)
        self.assertTrue(json.loads((out/"offline_preflight.json").read_text())["accepted"])
        self.assertFalse(json.loads((out/"replay_manifest.json").read_text())["frame_map"]["verified"])

    def test_mock_cli_exercises_the_device_worker_replay_protocol(self):
        episode=self.recording()
        script=Path(__file__).resolve().parents[1]/"scripts/replay_replica_trajectory.py"
        spec=importlib.util.spec_from_file_location("replay_cli_mock",script)
        cli=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        out=self.root/"mock_replay"
        code=cli.main(["--source",str(episode),"--hardware-config",str(self.config),
                       "--mode","mock","--output",str(out),"--no-video","--start-source-step","2"])
        self.assertEqual(code,0)
        summary=json.loads((out/"hardware/replay_summary.json").read_text())
        self.assertEqual(summary["status"],"replayed")
        self.assertEqual(summary["device_mode"],"mock")
        self.assertEqual(summary["real_api_calls"],0)
        import csv
        tracking=list(csv.DictReader((out/"hardware/replay_tracking.csv").open()))
        self.assertEqual(min(int(r["source_step"]) for r in tracking),2)
        config=json.loads((out/"hardware_config.json").read_text())
        self.assertEqual(config["reset_mode"],"none")


if __name__=="__main__":
    unittest.main()

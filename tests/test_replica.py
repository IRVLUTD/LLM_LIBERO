"""Source isolation, physical scoring, and the shared runner's audited exports."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import httpx
import numpy as np
from inspect_robots.types import Action
from inspect_robots_libero.evaluation_report import report_rows, write_reports
from inspect_robots_libero.replica.scenes import default_scene, compile_scene, prepare_scenes, sha256

ROOT = Path(__file__).resolve().parents[1]
SIM_AVAILABLE = bool(importlib.util.find_spec("mujoco") and importlib.util.find_spec("scipy"))
os.environ.setdefault("MUJOCO_GL", "egl")
INSTRUCTIONS = {1: "Put the bread on the red plate.", 2: "Put the blue bowl on the green coaster.",
    3: "Stack the blue block on the blue block.", 4: "Fold the pink towel in half.", 5: "Open the oven.",
    6: "Clean the whiteboard with the whiteboard eraser.", 7: "Pour 2 shakes of pepper into the red plate.",
    8: "Lift the blue bowl two times.", 9: "Press the button two times.", 10: "Collect 3 blocks into the yellow box."}


def load_runner():
    spec = importlib.util.spec_from_file_location("replica_runner", ROOT/"scripts/run_replica_evaluation.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


@unittest.skipUnless(SIM_AVAILABLE, "Install .[replica] for physical integration tests")
class ReplicaPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def config(self, task_id, variant=1):
        scene = default_scene(task_id, "ID", variant, INSTRUCTIONS[task_id])
        path = self.root/f"task{task_id:02d}_v{variant}"
        path.mkdir()
        config = path/"scene.json"
        config.write_text(json.dumps(scene))
        compile_scene(config)
        return config

    def env(self, config):
        from inspect_robots_libero.replica.embodiment import ReplicaEmbodiment, replica_task
        env = ReplicaEmbodiment(output_dir=self.root/(config.parent.name+"_run"), video=False, max_steps=80)
        self.addCleanup(env.close)
        return env, env.reset(replica_task(str(config)).scenes[0])

    def test_all_task_families_start_unsolved_and_control_tick_is_physical(self):
        for task_id in INSTRUCTIONS:
            with self.subTest(task_id=task_id):
                env, obs = self.env(self.config(task_id))
                start = obs.state["eef_state"].copy()
                for fraction in np.linspace(0, 1, 35):
                    target = start.copy()
                    target[2] += .015*fraction
                    obs = env.step(Action(data=target)).observation
                self.assertEqual(env.sim.steps, 35)
                self.assertGreater(obs.state["eef_state"][2]-start[2], .008)
                self.assertFalse(env.last["success"])
                self.assertFalse(any(w.number for w in env.sim.data.warning))
                self.assertEqual(set(obs.state), {"eef_state", "joint_angles", "gripper_measured", "controller_feedback"})
                env.close()

    def test_import_preserves_variant_reference_order_and_never_writes_source(self):
        source = self.root/"external"
        variants = source/"ID/task_variants/task02.json"
        variants.parent.mkdir(parents=True)
        instructions = [INSTRUCTIONS[2]]*5
        variants.write_text(json.dumps({"tasks": [{"name": s} for s in instructions]}))
        refs = source/"ID/referencePics/task02"
        refs.mkdir(parents=True)
        for i in range(1, 6):
            (refs/f"pic{i}.jpg").write_bytes(f"reference-{i}".encode())
        before = {str(p): sha256(p) for p in source.rglob("*") if p.is_file()}
        output = self.root/"catalog"
        catalog = prepare_scenes(source, output)
        self.assertEqual(len(catalog["scenes"]), 5)
        for i, entry in enumerate(catalog["scenes"], 1):
            scene = json.loads((output/entry["config"]).read_text())
            self.assertEqual(scene["variation"], i)
            self.assertTrue(scene["provenance"]["reference_image"].endswith(f"pic{i}.jpg"))
            self.assertEqual(scene["instruction"], instructions[i-1])
        self.assertEqual(before, {str(p): sha256(p) for p in source.rglob("*") if p.is_file()})
        with self.assertRaises(FileExistsError):
            prepare_scenes(source, output)

    def test_geometry_edit_changes_model_and_airborne_placement_does_not_score(self):
        import mujoco
        config = self.config(2)
        scene = json.loads(config.read_text())
        scene["objects"][0]["pos"][0] = .19
        config.write_text(json.dumps(scene))
        compile_scene(config)
        env, _ = self.env(config)
        self.assertAlmostEqual(env.sim.model.body("source").pos[0], .19)
        joint = env.sim.model.joint("source_free")
        adr = joint.qposadr[0]
        env.sim.data.qpos[adr:adr+3] = env.sim.data.body("target").xpos + [0, 0, .1]
        env.sim.gripper_command = 0.
        mujoco.mj_forward(env.sim.model, env.sim.data)
        for _ in range(15):
            self.assertFalse(env.scorer.update())
        # Place it on the coaster with physics, then require sustained support/release.
        env.sim.data.qpos[adr:adr+3] = env.sim.data.body("target").xpos + [0, 0, .009]
        env.sim.data.qvel[:] = 0.
        mujoco.mj_forward(env.sim.model, env.sim.data)
        env.sim.gripper_command = 0.
        for _ in range(30):
            env.sim.step()
        self.assertTrue(env.scorer.success)

    def test_counted_task_requires_full_cycles_and_rejects_overshoot(self):
        import mujoco
        env, _ = self.env(self.config(9))
        joint = env.sim.model.joint("button_slide")
        def depth(value):
            env.sim.data.qpos[joint.qposadr[0]] = -value
            mujoco.mj_forward(env.sim.model, env.sim.data)
            env.scorer.update()
        # Contact itself is tested by MuJoCo placement tests; isolate cycle logic here.
        with patch.object(env.scorer, "touching", return_value=True):
            depth(.008)
            for _ in range(20):
                depth(.009)
            self.assertEqual(env.scorer.count, 0)
            depth(.001)
            self.assertEqual(env.scorer.count, 1)
            self.assertFalse(env.scorer.success)
            depth(.008)
            depth(.001)
            for _ in range(10):
                depth(.001)
            self.assertTrue(env.scorer.success)
            depth(.008)
            self.assertFalse(env.scorer.success)
            depth(.001)
            self.assertEqual(env.scorer.count, 3)
            self.assertFalse(env.scorer.success)

    def test_real_policy_mock_requests_all_efforts_and_spreadsheet(self):
        runner = load_runner()
        config = self.config(2)
        out = self.root/"evaluation"
        sent = []
        original = runner.mock_transport
        def capture(model, wire):
            transport = original(model, wire)
            def reply(request):
                sent.append(json.loads(request.content))
                return transport.handle_request(request)
            return httpx.MockTransport(reply)
        # No real HTTP transport may be constructed during an offline run.
        with patch.object(runner, "mock_transport", side_effect=capture), patch.object(runner.httpx, "HTTPTransport", side_effect=AssertionError("network forbidden")):
            runner.main(["--scene-json", str(config), "--output", str(out), "--model", "openai/gpt-6-luna", "--mock", "--no-video"])
        data = json.loads((out/"evaluation.json").read_text())
        rows, calls, summaries = report_rows(out, data)
        self.assertEqual(len(rows), 3)
        self.assertEqual({body["reasoning"]["effort"] for body in sent}, {"low", "medium", "high"})
        self.assertEqual({r["status"] for r in rows}, {"finished"})
        self.assertEqual(len({r["initial_observation_sha256"] for r in rows}), 1)
        self.assertTrue(all(r["llm_calls"] == 2 and r["real_api_calls"] == 0 for r in rows))
        self.assertTrue(all(c["estimated_cost_usd"] is None and c["simulated_cost_usd"] > 0 for c in calls))
        with ZipFile(out/"evaluation.xlsx") as book:
            for name in book.namelist():
                if name.endswith(".xml") or name.endswith(".rels"):
                    ET.fromstring(book.read(name))
            self.assertEqual(len(ET.fromstring(book.read("xl/workbook.xml")).findall(".//{*}sheet")), 7)

    def test_unknown_usage_stops_entire_sweep(self):
        runner = load_runner()
        config = self.config(2)
        out = self.root/"unknown"
        def missing_usage(model, wire):
            return httpx.MockTransport(lambda request: httpx.Response(200, json={"id": "mock", "output": []}))
        with patch.object(runner, "mock_transport", side_effect=missing_usage):
            runner.main(["--scene-json", str(config), "--output", str(out), "--model", "openai/gpt-6-luna", "--mock", "--no-video"])
        data = json.loads((out/"evaluation.json").read_text())
        self.assertEqual(data["status"], "stopped_usage_unknown")
        self.assertEqual([r["status"] for r in data["runs"]], ["usage_unknown", "not_run", "not_run"])
        _, calls, _ = report_rows(out, data)
        self.assertIsNone(calls[0]["simulated_cost_usd"])


class ExportTests(unittest.TestCase):
    def test_unknown_cost_is_blank_not_zero_and_reasoning_cost_is_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root/"run/api"
            folder.mkdir(parents=True)
            run = dict(id="x", run_folder="run", model="openai/test", effort="high", benchmark="replica",
                       accounting_mode="paid", status="usage_unknown", usage_unknown=True, steps=20)
            data = dict(protocol={"pricing_per_million": {"openai/test": [1, .1, 1.25, 10]}}, runs=[run])
            events = [dict(event="request_started", call=1), dict(event="response", call=1, estimated_usd=.01,
                      usage=dict(input_tokens=100, output_tokens=100, output_tokens_details=dict(reasoning_tokens=40))),
                      dict(event="request_started", call=2), dict(event="usage_unknown", call=2)]
            (folder/"api_usage.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            episodes, calls, summaries = report_rows(root, data)
            self.assertIsNone(episodes[0]["estimated_cost_usd"])
            self.assertEqual(episodes[0]["known_cost_subtotal_usd"], .01)
            self.assertAlmostEqual(calls[0]["reasoning_output_cost_usd"], .0004)
            self.assertIsNone(calls[1]["estimated_cost_usd"])
            self.assertIsNone(summaries[0]["total_cost_usd"])


if __name__ == "__main__":
    unittest.main()

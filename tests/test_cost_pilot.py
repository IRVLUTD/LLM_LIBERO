"""Exercise the complete sweep with a fake simulator and mocked HTTP only."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import httpx

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_cost_pilot.py'
spec=importlib.util.spec_from_file_location('cost_pilot',SCRIPT)
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


class FakeEnv:
    def __init__(self, **kwargs):
        self.info=SimpleNamespace(action_space=object())
        self.last={'steps':1,'success':False}
        self.trajectory=[]
        self.initial_hash='same'
        self.episode_dir=None
    def close(self): pass


class PilotTests(unittest.TestCase):
    def run_pilot(self, root, reply, budget):
        def policy(**kwargs):
            return SimpleNamespace(transport=kwargs['transport'], model=kwargs['model'],
                                   _client=SimpleNamespace(close=lambda:None))
        def evaluate(task, pol, env, **kwargs):
            with httpx.Client(transport=pol.transport) as c:
                c.post('https://api.openai.com/v1/responses',json={
                    'model':pol.model.partition('/')[2], 'input':'test','tools':[]})
            return (SimpleNamespace(status='success'),)
        args=['pilot','--output',str(root),'--total-budget',str(budget),'--execute-paid-rollouts']
        with patch.object(pilot.sys,'argv',args), patch.object(pilot,'read_dotenv',return_value={'OPENAI_API_KEY':'fake'}), \
             patch.object(pilot,'RecordedEmbodiment',FakeEnv), patch.object(pilot,'agent_policy',policy), \
             patch.object(pilot,'evaluate',evaluate), patch.object(pilot,'ClampApprover',lambda _:None), \
             patch.object(pilot,'DeltaLimitApprover',lambda _:None), patch.object(pilot,'ChainApprover',lambda *a:None), \
             patch.object(pilot.httpx,'HTTPTransport',lambda **kw:httpx.MockTransport(reply)):
            pilot.main()
        return json.loads((root/'pilot.json').read_text())

    def test_shared_budget_and_all_configurations(self):
        sent=[]
        def reply(req):
            sent.append(req)
            return httpx.Response(200,json={'usage':{'input_tokens':100,'output_tokens':100},'output':[]})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'pilot'
            data=self.run_pilot(root,reply,.01)
            self.assertEqual(len(data['runs']),9)
            self.assertEqual(len(sent),3)  # Luna only; Sol/Astra full reservations cannot fit.
            self.assertLessEqual(data['budget_charge_usd'],.01)
            self.assertTrue(all(r['status']=='budget_stopped' for r in data['runs'][3:]))
            self.assertTrue((root/'report.html').exists())

    def test_unknown_usage_stops_entire_sweep(self):
        sent=[]
        def reply(req):
            sent.append(req)
            return httpx.Response(200,json={'output':[]})
        with tempfile.TemporaryDirectory() as tmp:
            data=self.run_pilot(Path(tmp)/'pilot',reply,3)
            self.assertEqual(len(sent),1)
            self.assertEqual(data['status'],'stopped_usage_unknown')
            self.assertGreater(data['budget_charge_usd'],0)
            self.assertTrue(all(r['status']=='not_run' for r in data['runs'][1:]))


if __name__=='__main__': unittest.main()

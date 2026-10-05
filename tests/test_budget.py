"""BudgetTransport: all HTTP is mocked, no key or .env needed."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import httpx
from inspect_robots_libero import budget
from inspect_robots_libero.budget import BudgetStop, BudgetTransport

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.sent=[]
        def reply(request):
            self.sent.append(json.loads(request.content))
            return httpx.Response(200,headers={'x-request-id':'req_mock'},json={
                'id':'resp_mock','usage':{'input_tokens':100,'output_tokens':50,
                'input_tokens_details':{'cached_tokens':20},
                'output_tokens_details':{'reasoning_tokens':30}},'output':[]})
        self.transport=BudgetTransport(self.tmp.name,15,inner=httpx.MockTransport(reply),model='openai/gpt-6-astra')
        self.client=httpx.Client(transport=self.transport)
    def tearDown(self):
        self.client.close(); self.transport.dispose(); self.tmp.cleanup()
    def request(self):
        return self.client.post('https://api.openai.com/v1/responses',json={
            'model':'gpt-6-astra','input':'offline','tools':[]})
    def test_requires_episode(self):
        with self.assertRaises(BudgetStop): self.request()
        self.assertEqual(self.sent,[])
    def test_budget_blocks_before_send(self):
        self.transport.begin_episode(0); self.transport.total_budget=.001
        with self.assertRaises(BudgetStop): self.request()
        self.assertEqual(self.sent,[])
    def test_accounting_caps_and_no_headers_logged(self):
        self.transport.begin_episode(0); self.request()
        self.assertEqual(self.sent[0]['max_output_tokens'],2048)
        self.assertEqual(self.sent[0]['service_tier'],'default')
        rows=[json.loads(x) for x in (Path(self.tmp.name)/'api_usage.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['request_id'],'req_mock')
        self.assertAlmostEqual(rows[-1]['estimated_usd'],.00332)
        self.assertNotIn('Authorization',json.dumps(rows))
    def test_full_output_cap_reserved_before_send(self):
        self.transport.begin_episode(0)
        self.transport.total_budget=.10
        with self.assertRaises(BudgetStop): self.request()
        self.assertEqual(self.sent, [])

    def test_configurable_output_cap(self):
        self.transport.max_output_tokens=8192
        self.transport.begin_episode(0)
        self.request()
        self.assertEqual(self.sent[0]['max_output_tokens'],8192)
        rows=[json.loads(x) for x in (Path(self.tmp.name)/'api_usage.jsonl').read_text().splitlines()]
        self.assertGreaterEqual(rows[0]['reservation_usd'],8192*50/1e6)

    def test_missing_output_usage_keeps_reservation_and_stops(self):
        self.transport.begin_episode(0)
        self.transport.inner=httpx.MockTransport(lambda r:httpx.Response(200,json={'usage':{'input_tokens':100}}))
        self.request()
        self.assertTrue(self.transport.uncertain)
        self.assertGreater(self.transport.used,0)
        with self.assertRaises(BudgetStop): self.request()

    def test_attempt_cap(self):
        self.transport.begin_episode(0); self.transport.calls=20
        with self.assertRaises(BudgetStop): self.request()
        self.assertEqual(self.sent,[])
    def test_unknown_usage_stops_subsequent_calls(self):
        self.transport.begin_episode(0)
        self.transport.inner=httpx.MockTransport(lambda r:httpx.Response(200,json={'output':[]}))
        self.request()
        with self.assertRaises(BudgetStop): self.request()
        self.assertTrue(self.transport.uncertain)
    def test_server_error_retried_as_one_call(self):
        self.transport.begin_episode(0)
        replies=iter([httpx.Response(503,json={'error':'mock'}),httpx.Response(529,json={'error':'mock'})])
        inner=self.transport.inner
        self.transport.inner=httpx.MockTransport(
            lambda r:next(replies,None) or inner.handle_request(r))
        with mock.patch.object(budget,'_SERVER_ERROR_BACKOFF_S',(0,0,0)):
            self.assertEqual(self.request().status_code,200)
        self.assertEqual(self.transport.calls,1)
        self.assertFalse(self.transport.uncertain)
    def test_exhausted_server_error_releases_reservation(self):
        self.transport.begin_episode(0)
        self.transport.inner=httpx.MockTransport(lambda r:httpx.Response(503,json={'error':'mock'}))
        with mock.patch.object(budget,'_SERVER_ERROR_BACKOFF_S',(0,)):
            self.assertEqual(self.request().status_code,503)
            self.assertEqual(self.request().status_code,503)
        self.assertFalse(self.transport.uncertain)
        self.assertAlmostEqual(self.transport.used,0)
    def test_forbid_other_models(self):
        self.transport.begin_episode(0)
        with self.assertRaises(BudgetStop):
            self.client.post('https://api.openai.com/v1/responses',json={'model':'gpt-6-sol','input':'x'})
        self.assertEqual(self.sent,[])
    def test_unpriced_model_rejected(self):
        with self.assertRaises(ValueError):
            BudgetTransport(self.tmp.name,15,model='openai/some-unpriced-model')
    def test_forbid_other_hosts(self):
        self.transport.begin_episode(0)
        with self.assertRaises(BudgetStop):
            self.client.post('https://example.com/v1/responses',json={'model':'gpt-6-astra'})
        self.assertEqual(self.sent,[])

class AnthropicBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.sent=[]
        def reply(request):
            self.sent.append((request.headers.get('anthropic-beta'),json.loads(request.content)))
            return httpx.Response(200,headers={'request-id':'req_mock'},json={
                'id':'msg_mock','type':'message','role':'assistant','content':[],
                'usage':{'input_tokens':100,'output_tokens':50,
                         'cache_creation_input_tokens':0,'cache_read_input_tokens':0}})
        self.transport=BudgetTransport(self.tmp.name,15,inner=httpx.MockTransport(reply),
                                       model='anthropic/claude-opus-5-5')
        self.client=httpx.Client(transport=self.transport)
    def tearDown(self):
        self.client.close(); self.transport.dispose(); self.tmp.cleanup()
    def test_messages_accounting(self):
        self.transport.begin_episode(0)
        self.client.post('https://api.anthropic.com/v1/messages',json={
            'model':'claude-opus-5-5','max_tokens':1000,'messages':[],'tools':[]})
        rows=[json.loads(x) for x in (Path(self.tmp.name)/'api_usage.jsonl').read_text().splitlines()]
        self.assertAlmostEqual(rows[-1]['estimated_usd'],(100*4.+50*20.)/1e6)
    def test_stale_thinking_blocks_dropped(self):
        self.transport.begin_episode(0)
        self.client.post('https://api.anthropic.com/v1/messages',headers={'anthropic-beta':'fast-mode-2026-02-01'},
                         json={'model':'claude-opus-5-5','max_tokens':1000,'messages':[],'tools':[],
                               'thinking':{'type':'adaptive'}})
        beta,body=self.sent[-1]
        self.assertEqual(body['thinking']['block_binding'],{'prefix_mismatch_behavior':'drop_block'})
        self.assertEqual(beta,'fast-mode-2026-02-01,'+budget._THINKING_BINDING_BETA)
    def test_thinking_signatures_not_counted_as_input(self):
        self.transport.begin_episode(0)
        thinking={'type':'thinking','thinking':'','signature':'x'*700000}
        self.client.post('https://api.anthropic.com/v1/messages',json={
            'model':'claude-opus-5-5','max_tokens':1000,'tools':[],
            'messages':[{'role':'assistant','content':[thinking]}]})
        _,body=self.sent[-1]
        self.assertEqual(len(body['messages'][0]['content'][0]['signature']),700000)
        rows=[json.loads(x) for x in (Path(self.tmp.name)/'api_usage.jsonl').read_text().splitlines()]
        self.assertLess(rows[0]['input_token_reservation'],10000)
    def test_openai_endpoint_rejected(self):
        self.transport.begin_episode(0)
        with self.assertRaises(BudgetStop):
            self.client.post('https://api.openai.com/v1/responses',json={'model':'claude-opus-5-5'})

if __name__=='__main__': unittest.main()

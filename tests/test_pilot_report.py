import json
from pathlib import Path
import tempfile
import unittest
from inspect_robots_libero.pilot_report import report_data, write_report


class ReportTests(unittest.TestCase):
    def test_truncation_unknown_usage_and_script_escaping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'pilot.json').write_text(json.dumps({'protocol':{},'runs':[
                {'id':'astra_high','api_attempts':1,'usage_unknown':False},
                {'id':'sol_low','api_attempts':1,'usage_unknown':True}]}))
            folder=root/'astra_high/api';folder.mkdir(parents=True)
            (folder/'api_usage.jsonl').write_text(json.dumps({'event':'response','call':1,
                'estimated_usd':.5,'usage':{'input_tokens':100,'output_tokens':8192}})+'\n')
            (folder/'ep00_call01_response.json').write_text(json.dumps({
                'incomplete_details':{'reason':'max_output_tokens'},'output':[
                {'type':'function_call','name':'move_to','arguments':'</script><script>alert(1)</script>'}]}))
            data=report_data(root)
            self.assertEqual(data['runs'][0]['truncated_calls'],1)
            self.assertTrue(data['runs'][0]['usage_complete'])
            self.assertFalse(data['runs'][1]['usage_complete'])
            write_report(root)
            html=(root/'report.html').read_text()
            self.assertNotIn('</script><script>alert(1)',html)
            self.assertIn('\\u003c/script\\u003e',html)
            self.assertTrue((root/'comparison.csv').exists())


if __name__=='__main__': unittest.main()

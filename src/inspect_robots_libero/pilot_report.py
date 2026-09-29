"""Standalone HTML report for a cost pilot. No CDN, server, or API requests."""
import base64
import csv
import io
import json
from pathlib import Path


def report_data(root):
    root = Path(root)
    data = json.loads((root/'pilot.json').read_text())
    for row in data['runs']:
        folder = root/row['id']
        events_path = folder/'api/api_usage.jsonl'
        events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        starts = {e['call']: e for e in events if e['event']=='request_started'}
        calls = []
        for event in events:
            if event['event'] != 'response':
                continue
            call = dict(event)
            call['reservation'] = starts.get(call['call'], {})
            response = folder/'api'/f"ep00_call{call['call']:02d}_response.json"
            payload = json.loads(response.read_text()) if response.exists() else {}
            call['tools'] = [{k: item.get(k) for k in ('name','arguments','call_id')}
                             for item in payload.get('output', []) if item.get('type')=='function_call']
            call['text'] = '\n'.join(part.get('text','') for item in payload.get('output', [])
                                    if item.get('type')=='message' for part in item.get('content', []))
            call['truncated'] = (payload.get('incomplete_details') or {}).get('reason')=='max_output_tokens'
            calls.append(call)
        row['calls'] = calls
        row['events'] = events
        row['truncated_calls'] = sum(c['truncated'] for c in calls)
        row['termination_reasons'] = []
        row['transcript'] = []
        logs = sorted((folder/'inspect').glob('*.json')) if (folder/'inspect').exists() else []
        for path in logs:
            log = json.loads(path.read_text())
            row['eval_stats'] = log.get('stats')
            for sample in log.get('samples', []):
                row['termination_reasons'].extend(sample.get('termination_reasons', []))
                for trial in sample.get('policy_transcripts', []):
                    row['transcript'].extend(trial)
        video = root/row['video'] if row.get('video') else None
        if video and video.is_file():
            row['video_data'] = 'data:video/mp4;base64,' + base64.b64encode(video.read_bytes()).decode()
        row['usage_complete'] = len(calls)==row.get('api_attempts',0) and all(
            c.get('estimated_usd') is not None for c in calls) and not row.get('usage_unknown',False)
    return data


def write_report(root):
    root = Path(root)
    data = report_data(root)
    # Escaping '<' makes arbitrary model output safe inside a script element.
    payload = json.dumps(data).replace('<', '\\u003c').replace('>', '\\u003e').replace('&','\\u0026')
    (root/'report.html').write_text(HTML.replace('__PILOT_DATA__',payload))
    public = json.loads(json.dumps(data))
    for row in public['runs']:
        row.pop('video_data',None)
    (root/'report_data.json').write_text(json.dumps(public,indent=2))
    with (root/'comparison.csv').open('w', newline='') as f:
        fields=['id','model','effort','status','api_attempts','estimated_usd','budget_charge_usd',
                'wall_seconds','steps','environment_success','truncated_calls','usage_complete']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(public['runs'])


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LIBERO · Model cost pilot</title>
<style>
:root{--bg:#09121e;--panel:#122131;--ink:#eef4fa;--muted:#adc0d0;--line:#2c4052;--luna:#62d9b5;--sol:#76baff;--astra:#d7a5ff;--warning:#ffc978}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}main{max-width:1420px;margin:auto;padding:42px 28px}header{border-bottom:1px solid var(--line);padding-bottom:25px;margin-bottom:25px}h1{font-size:clamp(30px,4vw,48px);letter-spacing:-1.7px;margin:8px 0}h2{font-size:22px;margin:0 0 15px}h3{font-size:17px;margin:0 0 12px}.eyebrow{color:var(--luna);text-transform:uppercase;letter-spacing:3px;font-size:12px}.sub,small,.muted{color:var(--muted)}.sub{max-width:950px}a{color:#8dcaff}button,select,input{font:inherit;color:var(--ink);background:#172b40;border:1px solid #466077;border-radius:7px;padding:8px 12px}button{cursor:pointer}button:hover{background:#25425f}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}.card,.panel{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px}.card .value{font-size:30px;font-weight:650;letter-spacing:-1px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}.notice{border-left:3px solid var(--warning);background:#30291e;padding:14px 18px;border-radius:6px;color:#fce2b7;margin:20px 0}.chart{width:100%;min-height:260px}svg{display:block;width:100%;overflow:visible}svg text{font:12px system-ui;fill:var(--muted)}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:500}tr:hover td{background:#1a3045}.tag{font-size:12px;padding:3px 8px;border-radius:10px;background:#223b50;display:inline-block}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:13px;color:var(--muted)}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}.video{width:100%;background:#050a10;border-radius:8px;max-height:430px}.metric-row{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px}details{border-top:1px solid var(--line);padding:12px 0}summary{cursor:pointer;color:#bfe0fa}pre{white-space:pre-wrap;word-break:break-word;font:12px/1.6 ui-monospace,monospace;max-height:450px;overflow:auto;background:#0a1724;padding:16px;border-radius:8px}.matrix{display:grid;grid-template-columns:65px repeat(3,minmax(0,1fr));gap:8px}.cell{padding:10px;overflow-wrap:anywhere;border-radius:8px;background:#203447}.cell small{display:block}.cell strong{font-size:20px}.cell.label{background:transparent;color:var(--muted)}footer{margin:30px 0;color:var(--muted);font-size:12px}.range{width:180px}.empty{color:var(--muted);padding:30px 0}.badge{color:var(--luna)}@media(max-width:900px){.grid{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}main{padding:22px 14px}.panel{padding:16px}.cell strong{font-size:16px}.matrix{grid-template-columns:45px repeat(3,minmax(0,1fr));gap:4px}}@media print{body{background:white;color:black}.panel,.card{background:white;color:black}.toolbar{display:none}svg text{fill:#333}pre{background:#eee;color:#111}}
</style></head><body><main>
<header><div class="eyebrow">LIBERO-10 / real simulator / paid API pilot</div><h1>What does a robot rollout cost?</h1><p class="sub" id="subtitle"></p><div class="legend"><span><i class="dot" style="background:var(--luna)"></i>GPT-6 Luna</span><span><i class="dot" style="background:var(--sol)"></i>GPT-6 Sol</span><span><i class="dot" style="background:var(--astra)"></i>GPT-6 Astra</span></div></header>
<div id="notice" class="notice"></div>
<div class="cards" id="cards"></div><section class="panel"><h2>What this pilot tells us</h2><ul id="findings"></ul></section>
<div class="toolbar"><label>Model <select id="filter"><option value="all">All models</option><option value="luna">Luna</option><option value="sol">Sol</option><option value="astra">Astra</option></select></label><button id="csv">Download comparison CSV</button><button id="json">Download full report data</button><button onclick="window.print()">Print / save PDF</button></div>
<div class="grid"><section class="panel"><h2>Measured API cost</h2><p class="muted">Sum of returned usage × published rates. Missing or unknown cost is never shown as a free run.</p><div id="cost" class="chart"></div></section><section class="panel"><h2>Model × reasoning effort</h2><p class="muted">Cost per short rollout; each starts from the same scene.</p><div class="matrix" id="matrix"></div></section></div>
<div class="grid"><section class="panel"><h2>Token composition</h2><div class="legend"><span>Input</span><span style="color:#e1b976">Cached input</span><span style="color:#d7a5ff">Reasoning</span><span style="color:#62d9b5">Other output</span></div><div id="tokens" class="chart"></div><small>Cached input is included in total input. Reasoning is included in total output, not added twice.</small></section><section class="panel"><h2>Response latency</h2><p class="muted">Mean API round-trip seconds; simulator initialization is excluded.</p><div id="latency" class="chart"></div></section></div>
<section class="panel"><h2>All nine configurations</h2><div class="scroll" id="comparison"></div></section>
<div class="grid"><section class="panel"><h2>Longer-episode cost scenarios</h2><div class="toolbar"><label>Calls per episode <input id="nCalls" class="range" type="range" min="1" max="100" value="20"> <b id="nLabel">20</b></label></div><div id="projection" class="chart scroll"></div><p class="muted">Estimate = observed mean cost/call × chosen calls. Range uses the lowest and highest observed call costs; it is not a confidence interval. Longer histories, different actions, cache behavior, and truncated reasoning can change cost. No estimate is made for incomplete usage, truncated responses, or runs with no move_to command. Give-up calls still contribute to measured costs; a very short sample may underestimate a sustained rollout.</p></section><section class="panel"><h2>Budget accounting</h2><div id="budget" class="chart"></div><p class="muted">The shared allowance counts all input at the cache-write rate and reserves the full output limit before sending. Actual provider billing may differ from usage-based estimates. Unconfirmed requests retain their reservation and stop the pilot.</p></section></div>
<section class="panel"><div class="toolbar"><h2 style="margin:0">Inspect a rollout</h2><select id="run"></select></div><div id="runmeta" class="metric-row"></div><div class="grid"><div><video id="video" class="video" controls preload="metadata"></video><p class="muted">Left: agent view · Right: wrist view · video is simulation time at 20 fps.</p></div><div><h3>End-effector position</h3><div id="trajectory" class="chart"></div><h3>Gripper closure</h3><div id="gripper"></div></div></div><h3>Per-call measurements and tool commands</h3><div class="scroll" id="calltable"></div><div id="calldetails"></div><details><summary>Conversation, observations, and tool feedback</summary><pre id="transcript"></pre></details><details><summary>Recorded robot states</summary><pre id="states"></pre></details><details><summary>Framework timing and API audit events</summary><pre id="audit"></pre></details></section>
<section class="panel" style="margin-top:20px"><h2>Protocol & interpretation</h2><ul id="notes"></ul><p id="hashes"></p><h3>Prices per million tokens (USD)</h3><div class="scroll" id="prices"></div><p id="sources"></p><details><summary>Complete experiment settings</summary><pre id="protocol"></pre></details></section>
<footer>Local, offline report. Charts and videos are embedded; no external scripts, analytics, or API calls. Sample size: at most one partial episode per configuration. This is a cost pilot, not a task-success benchmark.</footer>
</main><script id="data" type="application/json">__PILOT_DATA__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('data').textContent),P=D.protocol;
const $=id=>document.getElementById(id),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money=x=>x==null?'—':'$'+Number(x).toFixed(4),num=x=>x==null?'—':Number(x).toLocaleString(undefined,{maximumFractionDigits:1});
const model=r=>r.model.split('-').at(-1),color=r=>({luna:'#62d9b5',sol:'#76baff',astra:'#d7a5ff'}[model(r)]),label=r=>model(r)+' / '+r.effort;
const sum=(a,f)=>a.reduce((s,x)=>s+(Number(f(x))||0),0),measured=r=>r.calls.filter(c=>c.estimated_usd!=null),valid=r=>r.usage_complete&&measured(r).length>0;
const total=r=>valid(r)?sum(measured(r),c=>c.estimated_usd):null,mean=r=>valid(r)?total(r)/measured(r).length:null;
const motion=r=>r.calls.some(c=>c.tools.some(t=>t.name==='move_to')),projectable=r=>valid(r)&&!r.truncated_calls&&motion(r);
const rows=()=>D.runs.filter(r=>$('filter').value==='all'||model(r)===$('filter').value);
function table(headers,rows){return '<table><thead><tr>'+headers.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function bars(id,rs,get,fmt=money){const vals=rs.map(get),max=Math.max(...vals.filter(v=>v!=null),0.000001),h=rs.length*38+35;$ (id).innerHTML='<svg role="img" aria-label="'+esc(id)+'" viewBox="0 0 620 '+h+'">'+rs.map((r,i)=>{let v=vals[i],y=i*38+8,w=v==null?0:Math.max(1,v/max*370);return '<text x="0" y="'+(y+17)+'">'+esc(label(r))+'</text><rect x="135" y="'+y+'" width="'+w+'" height="24" rx="4" fill="'+color(r)+'" opacity=".85"><title>'+esc(label(r)+' '+fmt(v))+'</title></rect><text x="'+(145+w)+'" y="'+(y+17)+'">'+esc(fmt(v))+'</text>'}).join('')+'<text x="135" y="'+(h-2)+'">0</text><text x="490" y="'+(h-2)+'">'+esc(fmt(max))+'</text></svg>'}
function tokenChart(rs){let groups=rs.map(r=>{let cs=measured(r),inp=sum(cs,c=>c.usage.input_tokens),cached=sum(cs,c=>c.usage.input_tokens_details?.cached_tokens),out=sum(cs,c=>c.usage.output_tokens),reason=sum(cs,c=>c.usage.output_tokens_details?.reasoning_tokens);return [Math.max(0,inp-cached),cached,reason,Math.max(0,out-reason)]}),max=Math.max(1,...groups.map(g=>sum(g,x=>x))),cs=['#76baff','#e1b976','#d7a5ff','#62d9b5'];$('tokens').innerHTML='<svg viewBox="0 0 620 '+(rs.length*38+20)+'">'+rs.map((r,i)=>{let x=135,y=i*38+8;return '<text y="'+(y+17)+'">'+esc(label(r))+'</text>'+groups[i].map((v,j)=>{let w=v/max*370,s='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="24" fill="'+cs[j]+'"><title>'+['Uncached input','Cached input','Reasoning output','Other output'][j]+': '+v+'</title></rect>';x+=w;return s}).join('')+'<text x="'+(x+7)+'" y="'+(y+17)+'">'+(valid(r)?num(sum(groups[i],x=>x)):'—')+'</text>'}).join('')+'</svg>'}
function lines(id,points,indices,names,colors,height=210){if(!points.length){$(id).innerHTML='<p class="empty">No robot trajectory recorded.</p>';return}let vs=points.flatMap(p=>indices.map(i=>p.eef[i])),lo=Math.min(...vs),hi=Math.max(...vs);if(hi-lo<.001){lo-=.001;hi+=.001}let max=Math.max(1,...points.map(p=>p.step)),X=p=>45+p.step/max*510,Y=v=>height-30-(v-lo)/(hi-lo)*(height-60);$(id).innerHTML='<svg viewBox="0 0 620 '+height+'">'+[0,.5,1].map(t=>{let v=lo+(hi-lo)*t,y=Y(v);return '<line x1="45" x2="555" y1="'+y+'" y2="'+y+'" stroke="#2c4052"/><text x="0" y="'+(y+4)+'">'+v.toFixed(3)+'</text>'}).join('')+indices.map((index,j)=>'<polyline fill="none" stroke="'+colors[j]+'" stroke-width="2" points="'+points.map(p=>X(p)+','+Y(p.eef[index])).join(' ')+'"/><text x="'+(65+j*135)+'" y="15" style="fill:'+colors[j]+'">'+names[j]+'</text>').join('')+'<text x="45" y="'+(height-5)+'">step 0</text><text x="470" y="'+(height-5)+'">step '+max+'</text></svg>'}
function draw(){let rs=rows();bars('cost',rs,total);bars('latency',rs,r=>r.calls.length?sum(r.calls,c=>c.latency_s)/r.calls.length:null,x=>x==null?'—':x.toFixed(2)+' s');tokenChart(rs);$('comparison').innerHTML=table(['Configuration','Status','Calls','Cost USD','Budget charge','Wall s','Sim steps','Success','Truncated'],rs.map(r=>[label(r),r.status+(r.termination_reasons.length?' / '+r.termination_reasons.join(', '):''),r.api_attempts??0,money(total(r)),money(r.budget_charge_usd),num(r.wall_seconds),r.steps??'—',r.environment_success==null?'—':r.environment_success?'Yes':'No',r.truncated_calls]));let n=+$('nCalls').value;$('nLabel').textContent=n;bars('projection',rs,r=>projectable(r)?mean(r)*n:null);$('projection').innerHTML+=table(['Configuration','At '+n+' calls','Observed-call range'],rs.map(r=>{let cs=measured(r).map(c=>c.estimated_usd),ok=projectable(r);return[label(r),ok?money(mean(r)*n):'No representative motion sample',ok?money(Math.min(...cs)*n)+' – '+money(Math.max(...cs)*n):'—']}));bars('budget',rs,r=>r.budget_charge_usd??null);}
function inspect(){let r=D.runs.find(r=>r.id===$('run').value);$('runmeta').innerHTML=[['Status',r.status],['API calls',r.api_attempts??0],['Cost',money(total(r))],['Simulated seconds',num((r.steps??0)/20)],['Stop reason',r.stop_detail||r.termination_reasons.join(', ')||'—']].map(([k,v])=>'<span><small>'+esc(k)+'</small><br><b>'+esc(v)+'</b></span>').join('');if(r.video_data){$('video').src=r.video_data;$('video').style.display='block'}else{$('video').removeAttribute('src');$('video').style.display='none'}lines('trajectory',r.trajectory||[],[0,1,2],['x (m)','y (m)','z (m)'],['#76baff','#62d9b5','#d7a5ff']);lines('gripper',r.trajectory||[],[9],['0 open → 1 closed'],['#ffc978'],145);$('calltable').innerHTML=table(['Call','USD','Input','Cached','Output','Reasoning','Latency s','Images','Response','Tools'],r.calls.map(c=>[c.call,money(c.estimated_usd),c.usage?.input_tokens??'—',c.usage?.input_tokens_details?.cached_tokens??0,c.usage?.output_tokens??'—',c.usage?.output_tokens_details?.reasoning_tokens??'—',num(c.latency_s),c.reservation.images??'—',c.truncated?'TRUNCATED':c.response_status||('HTTP '+c.http_status),c.tools.map(t=>t.name).join(', ')||'None']));$('calldetails').innerHTML=r.calls.map(c=>'<details><summary>Call '+c.call+' · '+esc(c.tools.map(t=>t.name).join(', ')||'No tool call')+'</summary><pre>'+esc(JSON.stringify({tools:c.tools,text:c.text,usage:c.usage,latency_s:c.latency_s,incomplete_details:c.incomplete_details,reservation:c.reservation,request_id:c.request_id,response_id:c.response_id,http_status:c.http_status},null,2))+'</pre></details>').join('');$('transcript').textContent=JSON.stringify(r.transcript,null,2);$('states').textContent=JSON.stringify(r.trajectory||[],null,2);$('audit').textContent=JSON.stringify({eval_stats:r.eval_stats,events:r.events},null,2)}
$('subtitle').textContent='Task '+P.task_id+' · '+P.instruction+' · initial state '+P.init_state+' · '+P.max_calls+' calls/configuration · '+D.status+' · '+P.created_utc;
let attempts=sum(D.runs,r=>r.api_attempts),trunc=sum(D.runs,r=>r.truncated_calls),tested=D.runs.filter(r=>r.api_attempts>0).length;
$('notice').textContent='One partial episode per configuration: these measurements estimate API spending, not task success rates. '+(trunc?trunc+' response(s) hit the output limit; their longer-run projections are withheld. ':'')+(D.runs.some(r=>r.usage_unknown)?'Some usage is unknown: measured cost is a lower bound and the pilot stopped. ':'')+'Later calls can cost more as history grows. The model knows this short call allowance.';
$('cards').innerHTML=[['Measured spend',money(D.estimated_usd),'Known usage only'],['Shared allowance',money(P.total_budget_usd),'Conservative charge '+money(D.budget_charge_usd)],['Configurations sampled',tested+' / 9',attempts+' real API attempts'],['Output truncations',trunc,'Limit '+P.max_output_tokens.toLocaleString()+' tokens/call']].map(([k,v,s])=>'<div class="card"><small>'+esc(k)+'</small><div class="value">'+esc(v)+'</div><small>'+esc(s)+'</small></div>').join('');
let early=D.runs.filter(r=>r.api_attempts>0&&r.api_attempts<P.max_calls),noMotion=D.runs.filter(r=>r.api_attempts>0&&!motion(r));$('findings').innerHTML=[P.prompt_update_note||null,tested+' configurations sampled with '+attempts+' paid requests; known token-based cost '+money(D.estimated_usd)+'.',early.length?'Early exits: '+early.map(r=>label(r)+' ('+r.api_attempts+' calls)').join(', ')+'. Compare cost per call as well as total cost.':'Every sampled configuration used the requested call allowance.',noMotion.length?'No movement command was sampled for '+noMotion.map(label).join(', ')+'. Its cost measures an early exit, not a normal robot-control rollout.':'Movement commands were sampled in every tested configuration.', 'Use the recorded costs to size a larger pilot; one short rollout cannot rank task-solving ability.'].filter(Boolean).map(x=>'<li>'+esc(x)+'</li>').join('');
$('matrix').innerHTML='<div></div>'+['low','medium','high'].map(e=>'<div class="cell label">'+e+'</div>').join('')+['luna','sol','astra'].map(m=>'<div class="cell label">'+m+'</div>'+['low','medium','high'].map(e=>{let r=D.runs.find(r=>model(r)===m&&r.effort===e);return '<div class="cell" style="border-top:3px solid '+color(r)+'"><strong>'+money(total(r))+'</strong><small>'+(r.api_attempts??0)+' calls · '+esc(r.status)+'</small></div>'}).join('')).join('');
$('run').innerHTML=D.runs.map(r=>'<option value="'+esc(r.id)+'">'+esc(label(r))+'</option>').join('');
$('notes').innerHTML=P.notes.map(t=>'<li>'+esc(t)+'</li>').join('');let hashes=D.runs.map(r=>r.initial_observation_sha256).filter(Boolean);$('hashes').textContent='Initial camera/state hashes: '+(hashes.length?new Set(hashes).size===1?'identical across all '+hashes.length+' initialized runs.':'DIFFER across runs; inspect recorded states.':'no runs initialized.');
$('prices').innerHTML=table(['Model','Input','Cached input','Cache write','Output'],Object.entries(P.pricing_per_million).map(([m,r])=>[m,...r.map(money)]));$('sources').innerHTML='Official pricing checked '+esc(P.pricing_verified_date)+': '+P.pricing_sources.map(u=>'<a href="'+esc(u)+'">'+esc(u.split('/').at(-1))+'</a>').join(' · ');$('protocol').textContent=JSON.stringify(P,null,2);
function download(name,type,body){let a=document.createElement('a'),url=URL.createObjectURL(new Blob([body],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('json').onclick=()=>{let d=JSON.parse(JSON.stringify(D));d.runs.forEach(r=>delete r.video_data);download('report_data.json','application/json',JSON.stringify(d,null,2))};
$('csv').onclick=()=>{let rows=[['configuration','status','calls','estimated_usd','budget_charge_usd','wall_seconds','steps','truncations'],...D.runs.map(r=>[label(r),r.status,r.api_attempts,total(r),r.budget_charge_usd,r.wall_seconds,r.steps,r.truncated_calls])];download('comparison.csv','text/csv',rows.map(r=>r.map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')).join('\n'))};$('filter').onchange=draw;$('nCalls').oninput=draw;$('run').onchange=inspect;draw();inspect();
</script></body></html>'''

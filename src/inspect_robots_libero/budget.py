"""Fail-closed spending caps for paid LLM rollouts, as an httpx transport.

``LLMAgentPolicy`` accepts ``transport=``; this one sits between the policy and
the provider. Before each request it reserves a conservative cost estimate and
refuses to send when the per-episode or total budget would be exceeded, when a
previous request's usage is unknown, or when the request targets anything but
the model's own provider endpoint. Every request/response body is saved for
audit; headers (the API key) never touch disk.

Only models listed in ``PRICING`` are accepted. Add a row there (and a wire in
``PROVIDER_WIRE`` for a new provider) to budget another model.
"""
import base64
import hashlib
import io
import json
import sys
import time
from pathlib import Path
import httpx
from PIL import Image

class BudgetStop(RuntimeError):
    pass

_RESERVE_OUTPUT_TOKENS=700
# Hard backstop on estimated per-request size, independent of dollar cost --
# protects against a single request becoming pathologically bloated. Real
# model context windows are far larger (gpt-6-luna: 1.05M tokens per
# OpenAI's 2026-09-22 announcement), and this conservative estimate already
# overshoots real tokens substantially, so raising it is safe.
_MAX_INPUT_RESERVE_TOKENS=600000

# Which real endpoint + request shape a wire uses. 'responses' is OpenAI's
# Responses API; 'messages' is Anthropic's native Messages API -- different
# host, path, and request/response schema.
WIRE_ENDPOINTS={
    'responses':('api.openai.com','/v1/responses'),
    'messages':('api.anthropic.com','/v1/messages'),
}
# Provider prefix (the part before '/' in an upstream model id) -> wire this
# transport audits for it.
PROVIDER_WIRE={'openai':'responses','anthropic':'messages'}

# Per-1M-token USD rates: (regular_input, cached_input, cache_write, output),
# keyed by the upstream model id (``provider/model``). Input/output rates are
# the providers' published prices as of 2026-09-22. Cache sub-rates are
# estimated (cache_write=1.25x input, cached=0.1x input) and only affect
# ``estimated_usd``; ``budget_charge_usd`` always bills all input at the
# cache-write rate, so the budget stays conservative either way.
PRICING={
    'openai/gpt-6-astra':(10.,1.,12.5,50.),
    'openai/gpt-6-luna':(0.10,0.01,0.125,0.50),
    'openai/gpt-6-sol':(2.,0.2,2.5,10.),
    'anthropic/claude-opus-5-5':(4.,0.4,5.,20.),
}


def wire_for(model):
    """Return the wire this transport audits for ``model`` (a ``provider/model`` id)."""
    if model not in PRICING:
        raise ValueError(f'no pricing for {model!r}; budgeted models: {sorted(PRICING)}. '
                         'Add a row to PRICING in inspect_robots_libero/budget.py first.')
    return PROVIDER_WIRE[model.partition('/')[0]]


class BudgetTransport(httpx.BaseTransport):
    def __init__(self, output, total_budget, episode_budget=3., inner=None, task_id=0, demo_manifest=None, max_calls=20, model='openai/gpt-6-astra', image_horizon=2):
        self.output=Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.task_id=task_id
        self.max_calls=max_calls
        # Two cameras per kept observation; None keeps the full image history.
        self.max_live_images=None if image_horizon is None else 2*image_horizon
        self.wire=wire_for(model)
        self.model=model
        # Upstream strips the provider prefix before sending (resolve_provider).
        self.wire_model=model.partition('/')[2]
        self.demo_message=None
        self.demo_datas=frozenset()
        if demo_manifest is not None:
            from inspect_robots_libero.demos import load_demonstrations
            _, self.demo_message, self.demo_datas=load_demonstrations(demo_manifest,task_id,wire=self.wire)
        self.total_budget=total_budget
        self.episode_budget=episode_budget
        self.used=0.
        self.episode_used=0.
        self.estimated=0.
        self.episode_estimated=0.
        self.calls=0
        self.episode=None
        self.blocked=None
        self.uncertain=False
        self.inner=inner if inner is not None else httpx.HTTPTransport(retries=0)

    def begin_episode(self, episode):
        self.episode=episode
        self.episode_used=0.
        self.episode_estimated=0.
        self.calls=0
        self.blocked=None

    def record(self, data):
        with (self.output/'api_usage.jsonl').open('a') as f:
            f.write(json.dumps({'task_id':self.task_id,'episode':self.episode,'time':time.time(),**data})+'\n')
            f.flush()

    def progress(self, text):
        """One human-readable line per LLM call on stderr, so a live run shows progress."""
        print(f'[episode {self.episode} | call {self.calls}/{self.max_calls}] {text}',
              file=sys.stderr, flush=True)

    def stop(self, reason):
        self.blocked=reason
        self.record({'event':'blocked_before_request','reason':reason})
        self.progress(f'stopped before sending: {reason}')
        raise BudgetStop(reason)

    def _inject_demo(self, body):
        import copy
        if self.wire=='responses':
            items=body['input']
            index=1 if items and items[0].get('role') in ('system','developer') else 0
            body['input']=items[:index]+[copy.deepcopy(self.demo_message)]+items[index:]
        else:
            # Anthropic hoists system out of messages entirely (see upstream
            # inspect_robots_agent/_anthropic.py:_translate_messages), so there is no leading
            # system/developer turn to skip here -- index 0 is always right
            # before the first real turn ("Goal: ...").
            body['messages']=[copy.deepcopy(self.demo_message)]+body['messages']

    def _estimate(self, body):
        images=0
        image_tokens=0
        def walk(value):
            nonlocal images, image_tokens
            if isinstance(value,dict):
                if self.wire=='responses' and value.get('type')=='input_image':
                    url=value.get('image_url','')
                    if not url.startswith('data:image/png;base64,'):
                        self.stop('unsupported_image')
                    data=url.split(',',1)[1]
                    im=Image.open(io.BytesIO(base64.b64decode(data)))
                    expected_size=(1152,848) if data in self.demo_datas else (256,256)
                    if im.size != expected_size:
                        self.stop('unexpected_image_size')
                    images+=1
                    image_tokens+=8192
                    value['detail']='high'
                    return {'type':'input_image','image_url':'[image]','detail':'high'}
                if self.wire=='messages' and value.get('type')=='image':
                    source=value.get('source') or {}
                    if source.get('type')!='base64' or source.get('media_type')!='image/png':
                        self.stop('unsupported_image')
                    data=source.get('data','')
                    im=Image.open(io.BytesIO(base64.b64decode(data)))
                    expected_size=(1152,848) if data in self.demo_datas else (256,256)
                    if im.size != expected_size:
                        self.stop('unexpected_image_size')
                    images+=1
                    # Anthropic bills about w*h/750 tokens per image; reserve 2x.
                    image_tokens+=2*(-(-im.size[0]*im.size[1]//750))
                    return {'type':'image','source':{'type':'base64','media_type':'image/png','data':'[image]'}}
                return {k:walk(v) for k,v in value.items()}
            if isinstance(value,list):
                return [walk(v) for v in value]
            return value
        text=walk(body)
        return text,images,image_tokens

    def handle_request(self, request):
        if self.uncertain:
            self.stop('previous_request_usage_unknown')
        if self.episode is None:
            self.stop('not_inside_rollout')
        if self.calls>=self.max_calls:
            self.stop('api_attempt_limit')
        expected_host,expected_path=WIRE_ENDPOINTS[self.wire]
        if (request.method, request.url.scheme, request.url.host, request.url.path) != (
                'POST','https',expected_host,expected_path):
            self.stop('endpoint_not_allowed')
        body=json.loads(request.read())
        if body.get('model')!=self.wire_model:
            self.stop('model_not_allowed')
        tools=body.get('tools', [])
        if self.wire=='responses':
            if any(t.get('type')!='function' for t in tools):
                self.stop('non_robot_function_tool')
        else:
            # Anthropic's flat tool shape has no "type" key at all (see
            # _anthropic.py:_translate_tools); well-formedness is name +
            # input_schema instead.
            if any('name' not in t or 'input_schema' not in t for t in tools):
                self.stop('non_robot_function_tool')
        if self.demo_message is not None:
            self._inject_demo(body)
        if self.wire=='responses':
            body.update(max_output_tokens=2048, service_tier='default', store=False)
            body['metadata']={'experiment':'llm_libero_10','task_id':str(self.task_id),'episode':str(self.episode)}
            output_cap=2048
        else:
            # _anthropic.py already sets max_tokens on the outgoing body;
            # Responses-specific fields (max_output_tokens, service_tier,
            # store, metadata's free-form shape) don't exist on this wire and
            # would risk an unexpected-field rejection if forced on here.
            output_cap=int(body.get('max_tokens') or 16000)
        text,images,image_tokens=self._estimate(body)
        # Conservative local reservation, NOT a provider-guaranteed tokenizer.
        # Count all nonimage JSON bytes + per-image tokens (see _estimate) + schema overhead.
        # No cache discount in reservation; assume cache writes at 1.25x input price.
        input_reserve=len(json.dumps(text,ensure_ascii=False).encode())+image_tokens+4096
        too_many_images=(self.max_live_images is not None
                         and images>self.max_live_images+len(self.demo_datas))
        if too_many_images or input_reserve>_MAX_INPUT_RESERVE_TOKENS:
            self.stop('input_reservation_limit')
        # Real output across every call observed this session (zero-shot through
        # medium effort) has stayed under 450 tokens; RESERVE_OUTPUT_TOKENS is a
        # budget-planning assumption with a >1.5x margin over that, not the actual
        # output-length cap sent to the API above (which stays generous so the
        # model always has real headroom to think and respond).
        _,_,write_rate,out_rate=PRICING[self.model]
        reserve=input_reserve*write_rate/1e6+_RESERVE_OUTPUT_TOKENS*out_rate/1e6
        if self.used+reserve>self.total_budget or self.episode_used+reserve>self.episode_budget:
            self.stop('budget_reservation_limit')
        self.calls+=1
        self.used+=reserve
        self.episode_used+=reserve
        content=json.dumps(body).encode()
        # Preserve authorization only in memory; never write headers to disk.
        headers=dict(request.headers)
        headers.pop('content-length',None)
        outgoing=httpx.Request('POST',request.url,headers=headers,content=content,extensions=request.extensions)
        ident=f'ep{self.episode:02d}_call{self.calls:02d}'
        self.record({'event':'request_started','call':self.calls,'reservation_usd':reserve,
                     'input_token_reservation':input_reserve,'sha256':hashlib.sha256(content).hexdigest()})
        (self.output/f'{ident}_request.json').write_bytes(content)
        self.progress('waiting for the model...')
        try:
            response=self.inner.handle_request(outgoing)
            response.read()
        except Exception:
            self.uncertain=True
            self.record({'event':'usage_unknown','call':self.calls,'reserved_usd':reserve})
            self.progress('network error, usage unknown; stopping')
            raise BudgetStop('network_error_usage_unknown') from None
        payload=None
        try:
            payload=response.json()
        except ValueError:
            pass
        usage=payload.get('usage') if isinstance(payload,dict) else None
        (self.output/f'{ident}_response.json').write_bytes(response.content)
        error=payload.get('error') if isinstance(payload,dict) else None
        error_code=(error.get('code') if self.wire=='responses' else error.get('type')) if isinstance(error,dict) else None
        request_id=response.headers.get('x-request-id') or response.headers.get('request-id')
        data={'event':'response','call':self.calls,'http_status':response.status_code,
              'request_id':request_id,
              'response_id':payload.get('id') if isinstance(payload,dict) else None,'usage':usage,
              'api_error_code':error_code}
        if usage and response.status_code==200:
            in_rate,cached_rate,write_rate,out_rate=PRICING[self.model]
            out=int(usage['output_tokens'])
            if self.wire=='responses':
                inp=int(usage['input_tokens'])
                cached=int((usage.get('input_tokens_details') or {}).get('cached_tokens',0))
                written=int((usage.get('input_tokens_details') or {}).get('cache_write_tokens',0))
                total_in=inp
                estimated=((inp-cached-written)*in_rate+cached*cached_rate+written*write_rate+out*out_rate)/1e6
            else:
                # Anthropic reports input_tokens as ONLY the fresh (non-cached)
                # portion; cache_creation/cache_read are separate additive
                # fields, not subsets of input_tokens (opposite of OpenAI's
                # Responses shape, where input_tokens is the inclusive total).
                fresh=int(usage.get('input_tokens',0))
                written=int(usage.get('cache_creation_input_tokens',0))
                cached=int(usage.get('cache_read_input_tokens',0))
                total_in=fresh+written+cached
                estimated=(fresh*in_rate+cached*cached_rate+written*write_rate+out*out_rate)/1e6
            # Budget counts all input at cache-write rate, even if discounted in billing.
            conservative=(total_in*write_rate+out*out_rate)/1e6
            self.used+=conservative-reserve
            self.episode_used+=conservative-reserve
            data.update(estimated_usd=estimated,budget_charge_usd=conservative,
                        total_budget_charge_usd=self.used)
            self.estimated+=estimated
            self.episode_estimated+=estimated
            self.progress(f'ok: {total_in} in / {out} out tokens, ~${estimated:.3f}; '
                          f'episode ~${self.episode_estimated:.2f}, run ~${self.estimated:.2f} '
                          f'(budget reserved ${self.used:.2f}/{self.total_budget:g})')
            if total_in>input_reserve or out>output_cap:
                self.uncertain=True
                data['reservation_exceeded']=True
        else:
            self.uncertain=True
            data['reserved_usd']=reserve
            message=error.get('message','') if isinstance(error,dict) else ''
            self.progress(f'HTTP {response.status_code} {error_code or ""}: {message[:200]}')
        self.record(data)
        return response

    def close(self):
        # Shared across per-episode clients; the run owner disposes it once.
        pass

    def dispose(self):
        self.inner.close()

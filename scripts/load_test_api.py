"""Real SSE load test. Client timings are NOT server spans or model TTFT.
Run from repo root: python -m scripts.load_test_api --url URL --output DIR
Requires httpx. A fresh output directory prevents overwriting measurements.
"""
import argparse
import asyncio
import codecs
import csv
import json
import math
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx


class SSEParser:
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.buffer = ''
        self.kind = 'message'
        self.data = []

    def feed(self, chunk):
        self.buffer += self.decoder.decode(chunk)
        events = []
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line = line.rstrip('\r')
            if not line:
                if self.data:
                    events.append((self.kind, '\n'.join(self.data)))
                self.kind, self.data = 'message', []
            elif line.startswith('event:'):
                self.kind = line[6:].lstrip(' ')
            elif line.startswith('data:'):
                self.data.append(line[5:].lstrip(' '))
        return events


def admission_counts(log_path, wall_start, wall_end):
    """Counts chat_admission outcomes the server logged (chat_timing lines)
    within [wall_start, wall_end]. The load test's own summary can't tell a
    request rejected by admission control from one the agent simply failed
    -- both just show up as `ok: False` -- so this cross-checks against the
    server's own record of what it actually did with each request."""
    if not log_path:
        return {}
    counts = {'queue_full': 0, 'queue_timeout': 0, 'admission_acquired': 0}
    marker = 'chat_timing '
    try:
        with open(log_path) as f:
            for line in f:
                idx = line.find(marker)
                if idx == -1:
                    continue
                try:
                    payload = json.loads(line[idx + len(marker):])
                except ValueError:
                    continue
                stage = payload.get('stage')
                wall_time = payload.get('wall_time')
                if stage in counts and wall_time is not None and wall_start <= wall_time <= wall_end:
                    counts[stage] += 1
    except FileNotFoundError:
        return {}
    return counts


def responded(row):
    """True if the service produced a well-formed SSE response (reached a
    terminal done or error event over HTTP 200), regardless of whether the
    agent itself answered. Gating warmup on `ok` conflated two different
    things: a dead/misconfigured service, vs. this one randomly-picked
    question happening to be one the agent fails on its own (e.g.
    forced_answer_invalid) -- the agent is expected to fail a real fraction
    of questions even with zero load, so that shouldn't block the whole run."""
    return row.get('http_status') == 200 and any(e['kind'] in ('done', 'error') for e in row.get('events', []))


def percentile(values, p):
    if not values:
        return None
    xs = sorted(values)
    i = (len(xs)-1)*p
    a, b = math.floor(i), math.ceil(i)
    return xs[a] + (xs[b]-xs[a])*(i-a)


def summarize(rows, elapsed, concurrency):
    good = [r['total_s'] for r in rows if r['ok']]
    return dict(concurrency=concurrency, requests=len(rows), successful=len(good),
                error_rate=1-len(good)/len(rows), wall_s=elapsed,
                completed_rps=len(rows)/elapsed, successful_rps=len(good)/elapsed,
                success_mean_s=statistics.fmean(good) if good else None,
                success_p50_s=percentile(good,.5), success_p95_s=percentile(good,.95),
                all_p95_s=percentile([r['total_s'] for r in rows],.95))


def samples(corpus, seed):
    papers = {p['paper_id']: p for p in map(json.loads,(corpus/'papers.jsonl').read_text().splitlines())}
    pools = {k: [] for k in ['extractive','free_form','yes_no','unanswerable']}
    for q in map(json.loads,(corpus/'questions.jsonl').read_text().splitlines()):
        if papers[q['paper_id']]['split'] != 'validation':
            continue
        kinds = {a['answer_type'] for a in q['answers']}
        if len(kinds)!=1 or any(a.get('has_non_text_evidence') for a in q['answers']):
            continue
        kind = next(iter(kinds))
        if kind not in pools or (kind!='unanswerable' and not all(a.get('evidence_para_ids') for a in q['answers'])):
            continue
        pools[kind].append(dict(question_id=q['question_id'], paper_id=q['paper_id'],
                               answer_type=kind, question=q['question'],
                               title=papers[q['paper_id']]['title']))
    rng = random.Random(seed)
    chosen = []
    # 28 unique questions; repetitions below sustain the 32-user tier.
    for k,n in zip(pools,[14,4,5,5]):
        chosen += rng.sample(sorted(pools[k],key=lambda x:x['question_id']),n)
    rng.shuffle(chosen)
    return chosen


async def request(client, url, sample, timeout):
    start=time.perf_counter()
    row=dict(question_id=sample['question_id'], expected_paper_id=sample['paper_id'],
             started_utc=datetime.now(timezone.utc).isoformat(), ok=False, http_status=None,
             first_byte_s=None, first_event_s=None, first_answer_event_s=None, events=[])
    parser=SSEParser()
    answer=False
    done=False
    error=False
    try:
        async with asyncio.timeout(timeout):
            # Title included: ordinary UI question would otherwise be ambiguous.
            prompt=f'In the paper "{sample["title"]}": {sample["question"]}'
            async with client.stream('POST',url+'/api/chat',json={'question':prompt}) as response:
                row['http_status']=response.status_code
                row['request_id']=response.headers.get('x-request-id')
                row['headers_s']=time.perf_counter()-start
                response.raise_for_status()
                if 'text/event-stream' not in response.headers.get('content-type',''):
                    raise ValueError('Response is not SSE (authentication/proxy page?)')
                async for chunk in response.aiter_bytes():
                    now=time.perf_counter()-start
                    if chunk and row['first_byte_s'] is None:
                        row['first_byte_s']=now
                    for kind,data in parser.feed(chunk):
                        if row['first_event_s'] is None:
                            row['first_event_s']=now
                        payload=json.loads(data)
                        row['events'].append(dict(kind=kind,seconds=now,data=payload))
                        if kind=='token' and payload.get('text'):
                            answer=True
                            if row['first_answer_event_s'] is None:
                                row['first_answer_event_s']=now
                        if kind=='error':
                            error=True
                        if kind=='done':
                            done=True
            row['ok']=done and answer and not error
            if not row['ok']:
                row['error']='SSE error, missing answer, or missing done'
    except Exception as exc:
        row['error']=f'{type(exc).__name__}: {exc}'
    row['total_s']=time.perf_counter()-start
    return row


async def run(args):
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    batch=samples(Path(args.corpus),args.seed)
    (out/'questions.json').write_text(json.dumps(batch,ensure_ascii=False,indent=2))
    (out/'manifest.json').write_text(json.dumps(vars(args),indent=2))
    url=args.url.rstrip('/')
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout),follow_redirects=False,
                                 limits=httpx.Limits(max_connections=64,max_keepalive_connections=64)) as client:
        try:
            r=await client.get(url+'/api/papers')
            (out/'preflight.json').write_text(json.dumps(dict(status=r.status_code,content_type=r.headers.get('content-type')),indent=2))
            r.raise_for_status()
            if not isinstance(r.json().get('papers'),list):
                raise ValueError('Missing paper catalog')
        except Exception as exc:
            (out/'blocked.json').write_text(json.dumps({'error':str(exc),'status':'blocked_before_load'}))
            print(f'BLOCKED: {exc}',flush=True)
            return
        warm=await request(client,url,batch[0],args.timeout)
        (out/'warmup.json').write_text(json.dumps(warm,ensure_ascii=False,indent=2))
        if not responded(warm):
            print('BLOCKED: warmup got no well-formed SSE response; no load submitted',flush=True)
            return
        if not warm['ok']:
            print(f"Warmup question itself failed ({warm.get('error')}), but the service "
                  f"responded -- proceeding with the load test.",flush=True)
        summaries=[]
        for c in args.concurrency:
            queue=asyncio.Queue()
            for repeat in range(args.repeats):
                for q in batch:
                    queue.put_nowait((repeat,q))
            rows=[]
            async def worker():
                while not queue.empty():
                    repeat,q=queue.get_nowait()
                    row=await request(client,url,q,args.timeout)
                    row.update(concurrency=c,repeat=repeat)
                    rows.append(row)
                    with (out/'requests.jsonl').open('a') as f:
                        f.write(json.dumps(row,ensure_ascii=False)+'\n')
            wall_start=time.time()
            start=time.perf_counter()
            await asyncio.gather(*(worker() for _ in range(c)))
            wall_end=time.time()
            summaries.append(summarize(rows,time.perf_counter()-start,c))
            summaries[-1].update(admission_counts(args.server_log,wall_start,wall_end))
            with (out/'summary.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=summaries[0].keys())
                writer.writeheader(); writer.writerows(summaries)
            print(json.dumps(summaries[-1]),flush=True)
            if summaries[-1]['error_rate']>=args.stop_error_rate:
                print('Stopping escalation: error-rate threshold reached',flush=True)
                break
            if c!=args.concurrency[-1]:
                await asyncio.sleep(args.gap)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--corpus',default='corpus/qasper')
    p.add_argument('--concurrency',type=int,nargs='+',default=[1,2,4,8,16,32])
    p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--gap',type=float,default=15)
    p.add_argument('--seed',type=int,default=599)
    p.add_argument('--stop-error-rate',type=float,default=.25)
    p.add_argument('--server-log',default=None,
                   help='Path to the backend process log (e.g. app_server.log), to cross-check '
                        'client-observed failures against the server\'s own chat_timing admission records.')
    a=p.parse_args()
    if a.repeats<1 or any(c<1 or c>32 for c in a.concurrency):
        p.error('repeats>=1 and concurrency between 1 and 32 required')
    asyncio.run(run(a))

if __name__=='__main__':
    main()

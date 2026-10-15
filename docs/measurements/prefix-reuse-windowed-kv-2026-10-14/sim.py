"""Replay of cuda/kvwindow.go's windowed-KV base arithmetic over the I3 night's Codex x Mellum2.1 requests.

Usage: python3 sim.py <records.jsonl[.gz]> <serve-mellum-16k.log[.gz]>
(the job-b files under docs/measurements/integration-harness-2026-10/i3-night-2026-10-13/).

For each /v1/responses request it takes the prompt and completion sizes from the run records and the server's time to first
token from the serve log (cold = over 20 s). It assumes one KV slot (the serve log says so), a follow-up inside a run that
diverges at the previous prompt's end (Codex sends no reasoning back), and, for a run's first request, a divergence 30 tokens
before its end (not recorded; either rule calls those cold). It prints whether the old rule (slack 512, no rollback kept)
predicts each request cold, against the server, and what the rule that keeps a rollback (kvWindowNextBase) predicts at
several budgets. The same replay runs as the Go unit test TestWindowedKV_i3Replay.
"""
import gzip, json, re, sys

W = 1024


def opener(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path)


def replay(reqs, slack, rollback):
    cap = W + slack
    chunk = min(512, slack + 1)
    base, held, out = 0, 0, []

    def write(p0, n):
        nonlocal base
        if p0 == 0:
            base = 0
        low = max(0, p0 - (W - 1))
        assert low >= base, (p0, base)
        if p0 + n - base > cap:
            base = max(base, min(low, p0 + n - (W - 1) - rollback))

    for i, (prompt, gen, obs, run) in enumerate(reqs):
        first = i == 0 or reqs[i - 1][3] != run
        lcp = prompt - 30 if first else min(reqs[i - 1][0], prompt - 1)
        lcp = min(lcp, held)
        reuse = lcp if (base == 0 or lcp - (W - 1) >= base) else 0
        out.append(dict(cold=reuse == 0, first=first, rollback=held - lcp, obs=obs, prompt=prompt, gen=gen, run=run))
        p = reuse
        while p < prompt:
            n = min(chunk, prompt - p)
            write(p, n)
            p += n
        for q in range(prompt, prompt + gen - 1):
            write(q, 1)
        held = prompt + gen - 1
    return out


def main():
    recs = [json.loads(l) for l in opener(sys.argv[1])]
    ttft = [int(m.group(1)) for m in re.finditer(r'POST /v1/responses .*ttft=(\d+)ms', opener(sys.argv[2]).read())]
    reqs = [(q['prompt_tokens'], q['completion_tokens'], r['id']) for r in recs if r['agent'] == 'codex' for q in r['wire']['per_request']]
    assert len(reqs) == len(ttft), (len(reqs), len(ttft))
    reqs = [(p, g, t > 20000, run) for (p, g, run), t in zip(reqs, ttft)]
    old = replay(reqs, 512, 0)
    for x in old:
        print('%-24s prompt %5d gen %5d rollback %5d  replay cold=%d  server cold=%d %s' % (
            x['run'], x['prompt'], x['gen'], x['rollback'], x['cold'], x['obs'], '' if x['cold'] == x['obs'] else 'MISMATCH'))
    print('old rule (slack 512, rollback 0): %d cold, server %d, agree %d of %d' % (
        sum(x['cold'] for x in old), sum(x['obs'] for x in old), sum(x['cold'] == x['obs'] for x in old), len(old)))
    for r in (512, 1024, 1536, 2048):
        new = replay(reqs, r + 512, r)
        inrun = [x for x in new if not x['first']]
        print('rollback %4d (slack %4d): %2d cold; in-run %d of %d cold, rolling back %s' % (
            r, r + 512, sum(x['cold'] for x in new), sum(x['cold'] for x in inrun), len(inrun),
            [x['rollback'] for x in inrun if x['cold']]))


if __name__ == '__main__':
    main()

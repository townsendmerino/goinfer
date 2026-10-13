#!/usr/bin/env python3
"""KV3 speed gate (docs/measurements/kv3-cuda-2026-10-13.md §2.3): time to first token for a Claude Code-sized first request, the
new serve binary against the old, three alternating pairs, a fresh server per run.

  cc_first_request_ab.py <old serve> <new serve> <model dir> <recorded Claude Code request body> <out json> [--ctx 28672] [--pairs 3]

The request is KV0's recorded Claude Code first request (17,755 tokens under Qwen3's template), replayed through /v1/messages with
max_tokens 1 and no streaming, so the time to the response is the time to the first token. Each run records the server's own
prompt_tokens, the swap in use before the load and after the answer (the never-swap reading: growth of 64 MB or under passes),
and which decode and prefill path the load resolved to. Holds the timing lock the way bench_peer.py does (scripts/timing_lock.py,
taken by the night queue for the job).
"""
import json, os, re, signal, subprocess, sys, time, urllib.request

old, new, model, body_path, out = sys.argv[1:6]
ctx = int(sys.argv[sys.argv.index("--ctx") + 1]) if "--ctx" in sys.argv else 28672
pairs = int(sys.argv[sys.argv.index("--pairs") + 1]) if "--pairs" in sys.argv else 3
PORT = 18192


def swap_used_mb():
    m = dict(l.split(":")[0:2] for l in open("/proc/meminfo"))
    kb = lambda k: int(m[k].split()[0])
    return (kb("SwapTotal") - kb("SwapFree")) // 1024


def run(arm, binary, i):
    log = f"{out}.{arm}{i}.serve.log"
    swap0 = swap_used_mb()
    p = subprocess.Popen([binary, "--model", model, "--backend", "cuda", "--ctx", str(ctx), "--kv-sessions", "1", "--quant", "int4",
                          "--addr", f"127.0.0.1:{PORT}", "--log-requests"], stdout=open(log, "w"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True, env={**os.environ, "GOINFER_SWAP_GUARD": "4096"})
    try:
        for _ in range(600):
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)).get("data"):
                    break
            except Exception:
                time.sleep(1)
        body = json.loads(json.load(open(body_path))["body"])
        body["max_tokens"], body["stream"] = 1, False
        body.pop("thinking", None)
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/messages", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "x-api-key": "x", "anthropic-version": "2023-06-01"})
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(req, timeout=1500))
            status, ptok = 200, (r.get("usage") or {}).get("input_tokens")
        except urllib.error.HTTPError as e:
            status, ptok = e.code, None
        dt = time.time() - t0
        swap1 = swap_used_mb()
        txt = open(log).read()
        path = re.findall(r"decode path: (.*)", txt)
        declined = len(re.findall(r"batched prefill declined", txt))
        rec = dict(arm=arm, run=i, status=status, seconds=round(dt, 1), prompt_tokens=ptok, swap_before_mb=swap0, swap_after_mb=swap1,
                   swap_growth_mb=swap1 - swap0, decode_path=path[:1], batched_prefill_declined_lines=declined)
        print(json.dumps(rec), flush=True)
        return rec
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        time.sleep(8)


recs = []
for i, order in enumerate([("old", "new"), ("new", "old"), ("old", "new")][:pairs], 1):
    for arm in order:
        recs.append(run(arm, old if arm == "old" else new, i))
        json.dump(recs, open(out, "w"), indent=1)
o = sorted(r["seconds"] for r in recs if r["arm"] == "old" and r["status"] == 200)
n = sorted(r["seconds"] for r in recs if r["arm"] == "new" and r["status"] == 200)
med = lambda x: x[len(x) // 2] if x else None
mo, mn = med(o), med(n)
verdict = "NO READING"
if mn is not None:
    verdict = "PASS" if mn <= 45 else ("AMBIGUOUS" if mn <= 90 else "FAIL")
swap = max((r["swap_growth_mb"] for r in recs if r["arm"] == "new"), default=None)
summary = dict(old_seconds=o, new_seconds=n, old_median=mo, new_median=mn, ratio=(round(mo / mn, 2) if mo and mn else None), verdict=verdict,
               never_swap_max_growth_mb_new_arm=swap, never_swap="PASS" if swap is not None and swap <= 64 else "FAIL")
print("SUMMARY", json.dumps(summary), flush=True)
json.dump(dict(runs=recs, summary=summary), open(out, "w"), indent=1)

#!/usr/bin/env python3
"""Per-token logprobs of the flipping prompts (copy sections 21, 0, 2, 9), every prompt COLD (a fresh server, each prompt asked once, so no KV reuse between runs).
mode solo: a fresh server per prompt, the prompt alone.  mode conc: ONE fresh server, the four prompts at once (STAGGER=s: request i starts i*s seconds after the first).
conc_logprobs.py's solo-vs-conc comparison was confounded: its second phase re-sent prompts the server had already seen, so it compared a cold prefill with a reused one.
  GOINFER_SERVE_CPU=<binary> BENCH_W7_MODEL=<gguf> python3 cold_logprobs.py OUT.json solo|conc [serve-args]"""
import json, os, sys, threading, time
sys.path.insert(0, "/home/francis/mycode/goinfer/scripts")
import bench_w7_plain as w7, bench_spec_copy as sc
OUT, MODE, SARGS = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
IDX = [21, 0, 2, 9]
secs = sc.sections("cc5f8c2c", 3500, 30)
def ask(srv, k):
    body = dict(sc.request(secs[k]), max_tokens=12, logprobs=True, top_logprobs=3)
    _, r = w7.post(srv.url, body, timeout=300)
    return {"reused": r["usage"].get("prefill_reused_tokens"), "ptoks": r["usage"]["prompt_tokens"],
            "lp": [(t["token"], t["logprob"]) for t in r["choices"][0]["logprobs"]["content"]]}
def warm(srv):
    if os.environ.get("NOWARM"): return   # NOWARM=1: no warm-up request, so no slot holds a prefix a prompt could reuse
    w7.post(srv.url, {"model": "bench", "messages": [{"role": "user", "content": "warm"}], "max_tokens": 4, "temperature": 0})
res = {}
if MODE == "solo":
    for k in IDX:
        with w7.GoinferServer("cuda", SARGS, "/dev/null") as srv:
            warm(srv); res[k] = ask(srv, k)
else:
    with w7.GoinferServer("cuda", SARGS, "/dev/null") as srv:
        warm(srv)
        def go(k):
            time.sleep(IDX.index(k) * float(os.environ.get("STAGGER", "0"))); res[k] = ask(srv, k)
        ths = [threading.Thread(target=go, args=(k,)) for k in IDX]; [t.start() for t in ths]; [t.join() for t in ths]
json.dump({str(k): v for k, v in res.items()}, open(OUT, "w"))
for k in IDX: print(f"{MODE} section {k:2d}: prompt_tokens={res[k]['ptoks']} reused={res[k]['reused']} first logprobs " + " ".join(f"{t[1]:.4f}" for t in res[k]["lp"][:6]))

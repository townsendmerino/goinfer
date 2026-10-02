#!/usr/bin/env python3
"""Where does concurrent serving perturb a reply? Per-token logprobs of the prompts that flip (copy sections 21, 0, 2, 9), solo (one request at a
time) against four at once, on the plain batch arm. A shift at token 0 is PREFILL numerics (token 0 is argmax of the prompt's last row); a shift
only from token 1 on is the DECODE step. Diagnostic, exploratory.   GOINFER_SERVE_CPU=<binary> BENCH_W7_MODEL=<gguf> python3 conc_logprobs.py OUT.json [serve-args]
"""
import json, os, sys, threading, time
sys.path.insert(0, "/home/francis/mycode/goinfer/scripts")
import bench_w7_plain as w7, bench_spec_copy as sc

OUT, SARGS = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
IDX = [21, 0, 2, 9]
NTOK = 12
secs = sc.sections("cc5f8c2c", 3500, 30)


def ask(srv, k):
    body = dict(sc.request(secs[k]), max_tokens=NTOK, logprobs=True, top_logprobs=3)
    _, resp = w7.post(srv.url, body, timeout=300)
    lp = resp["choices"][0]["logprobs"]["content"]
    return [{"tok": t["token"], "lp": t["logprob"], "top": [(x["token"], x["logprob"]) for x in t["top_logprobs"]]} for t in lp]


res = {"solo": {}, "conc": []}
with w7.GoinferServer("cuda", SARGS, OUT + ".servers.log") as srv:
    w7.post(srv.url, {"model": "bench", "messages": [{"role": "user", "content": "warm"}], "max_tokens": 4, "temperature": 0})
    for k in IDX:                      # solo, one at a time (the order shuffled once so no prompt always follows the same one)
        res["solo"][k] = ask(srv, k)
    for rep in range(3):               # four at once, three times
        got = {}

        def go(k):
            time.sleep(IDX.index(k) * float(os.environ.get("STAGGER", "0")))   # STAGGER=1: request i starts i s after the first
            got[k] = ask(srv, k)
        ths = [threading.Thread(target=go, args=(k,)) for k in IDX]
        [t.start() for t in ths]
        [t.join() for t in ths]
        res["conc"].append(got)
json.dump({"solo": {str(k): v for k, v in res["solo"].items()}, "conc": [{str(k): v for k, v in g.items()} for g in res["conc"]]}, open(OUT, "w"))

print(f"serve args: {SARGS!r}")
for rep, got in enumerate(res["conc"]):
    for k in IDX:
        a, b = res["solo"][k], got[k]
        same_tok = [x["tok"] == y["tok"] for x, y in zip(a, b)]
        first_flip = same_tok.index(False) if False in same_tok else None
        d = [abs(x["lp"] - y["lp"]) for x, y in zip(a, b)]
        print(f"  rep {rep} section {k:2d}: max|dlogprob| by token {[round(v, 4) for v in d]}  token flip at {first_flip}")

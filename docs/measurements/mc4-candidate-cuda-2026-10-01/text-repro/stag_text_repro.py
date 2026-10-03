#!/usr/bin/env python3
"""MC4 candidate: the STAGGERED identity divergence (reply 21 of 30 differs for the candidate in all 3 graded rounds), with reply TEXT kept so
the first divergence can be seen. Diagnostic, exploratory, not a measurement. Same pinned serve-cuda binary, a fresh server per run.
The staggered cell exactly as bench_spec_copy.py builds it: rounds 12,9,6,3, client i starts i*2 s after the cell begins, request = copy section
first[i]+r of decoder/model.go@cc5f8c2c, 256 max tokens, temperature 0.
  GOINFER_SERVE_CPU=<serve-cuda binary> BENCH_W7_MODEL=<1.5B gguf> python3 stag_text_repro.py OUTDIR
"""
import json, os, sys, threading, time
sys.path.insert(0, "/home/francis/mycode/goinfer/scripts")
import bench_w7_plain as w7
import bench_spec_copy as sc

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
ARMS = {"b": "", "b2": "", "s": "-spec=ngram", "c1": "-spec=ngram -spec-adaptive", "c2": "-spec=ngram -spec-adaptive", "c3": "-spec=ngram -spec-adaptive",
        "cny": "-spec=ngram -spec-adaptive"}
ARMS["c"] = ARMS["c1"]
ENVS = {"cny": {"GOINFER_SPEC_ADAPTIVE_NEVER_YIELD": "1"}}
ROUNDS = [int(x) for x in os.environ.get("ROUNDS", "12,9,6,3").split(",")]
STAGGER = float(os.environ.get("STAGGER", "2.0"))   # ROUNDS=30 STAGGER=0: one client, the same 30 prompts, no concurrency
FIRST = [sum(ROUNDS[:i]) for i in range(len(ROUNDS))]


def run(arm):
    os.environ.pop("GOINFER_SPEC_ADAPTIVE_NEVER_YIELD", None)
    os.environ.update(ENVS.get(arm, {}))
    secs = sc.sections("cc5f8c2c", 3500, sum(ROUNDS))
    res = {}
    with w7.GoinferServer("cuda", ARMS[arm], OUT + "/servers.log") as srv:
        w7.post(srv.url, {"model": "bench", "messages": [{"role": "user", "content": "warm"}], "max_tokens": 4, "temperature": 0})
        t0 = time.perf_counter()

        def client(i):
            time.sleep(i * STAGGER)
            for r in range(ROUNDS[i]):
                body = dict(sc.request(secs[FIRST[i] + r]), max_tokens=256)
                start = time.perf_counter() - t0
                lat, resp = w7.post(srv.url, body, timeout=600)
                c = resp["choices"][0]["message"]["content"] or ""
                res[(i, r)] = {"text": c, "ctoks": (resp.get("usage") or {}).get("completion_tokens"), "reused": (resp.get("usage") or {}).get("prefill_reused_tokens"), "start": round(start, 2), "lat": round(lat, 2)}
        ths = [threading.Thread(target=client, args=(i,)) for i in range(len(ROUNDS))]
        [t.start() for t in ths]
        [t.join() for t in ths]
    return res


def firstdiff(a, b):
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return None if len(a) == len(b) else n


allres = {}
for arm in sys.argv[2:] or list(ARMS):
    t0 = time.time()
    allres[arm] = run(arm)
    print(f"[{time.strftime('%H:%M:%S')}] {arm} done in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    json.dump({a: {f"{k[0]}/{k[1]}": v for k, v in r.items()} for a, r in allres.items()}, open(OUT + "/texts.json", "w"), indent=1)
base = allres["b"]
for arm, r in allres.items():
    if arm == "b":
        continue
    same = 0
    print(f"\n=== {arm} vs b ===")
    for k in sorted(base):
        fd = firstdiff(base[k]["text"], r[k]["text"])
        if fd is None:
            same += 1
        else:
            print(f"  client {k[0]} req {k[1]} (index {FIRST[k[0]] + k[1]}): first diff at char {fd} of {len(base[k]['text'])} (ctoks b {base[k]['ctoks']}, {arm} {r[k]['ctoks']}; started b {base[k]['start']}s, {arm} {r[k]['start']}s); "
                  f"b: ...{base[k]['text'][max(0,fd-25):fd+30]!r}  {arm}: ...{r[k]['text'][max(0,fd-25):fd+30]!r}")
    print(f"  {arm}: {same}/{len(base)} identical to batch")

#!/usr/bin/env python3
"""MC4 candidate identity bug: keep reply TEXT, find the first divergence. Diagnostic, exploratory, not a measurement.
Arms (same pinned serve-cuda binary, fresh server each): b=batch, b2=batch again (determinism), s=-spec ngram, c=-spec ngram -spec-adaptive.
Workloads at 4 clients: copy (2 requests per client, 256 tokens) and chat (one plain turn each, 128 tokens)."""
import json, os, sys, threading, time
sys.path.insert(0, "/home/francis/mycode/goinfer/scripts")
import bench_w7_plain as w7
import bench_spec_copy as sc

OUT = sys.argv[1]
ARMS = {"b": "", "b2": "", "s": "-spec=ngram", "c": "-spec=ngram -spec-adaptive"}
CHAT = [t for t in w7.TURNS[:4]]


def run(arm, workload):
    secs = sc.sections("cc5f8c2c", 3500, 8)
    res = {}
    with w7.GoinferServer("cuda", ARMS[arm], OUT + "/servers.log") as srv:
        w7.post(srv.url, {"model": "bench", "messages": [{"role": "user", "content": "warm"}], "max_tokens": 4, "temperature": 0})

        def client(i):
            reqs = []
            if workload == "copy":
                for r in range(2):
                    reqs.append(dict(sc.request(secs[i * 2 + r]), max_tokens=256))
            else:
                reqs.append({"model": "bench", "temperature": 0, "max_tokens": 128, "messages": [{"role": "user", "content": CHAT[i]}]})
            for r, body in enumerate(reqs):
                lat, resp = w7.post(srv.url, body, timeout=600)
                c = resp["choices"][0]["message"]["content"] or ""
                res[(i, r)] = {"text": c, "ctoks": (resp.get("usage") or {}).get("completion_tokens")}

        ths = [threading.Thread(target=client, args=(i,)) for i in range(4)]
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
for wl in ("copy", "chat"):
    for arm in ("b", "b2", "s", "c"):
        t0 = time.time()
        allres[(wl, arm)] = run(arm, wl)
        print(f"[{time.strftime('%H:%M:%S')}] {wl} {arm} done in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    base = allres[(wl, "b")]
    print(f"\n=== {wl}: compare to batch (b) ===")
    for arm in ("b2", "s", "c"):
        r = allres[(wl, arm)]
        same = 0
        for k in sorted(base):
            fd = firstdiff(base[k]["text"], r[k]["text"])
            if fd is None:
                same += 1
            else:
                print(f"  {arm} client {k[0]} req {k[1]}: first diff at char {fd} of {len(base[k]['text'])} (b ctoks {base[k]['ctoks']}, {arm} ctoks {r[k]['ctoks']}); "
                      f"b: ...{base[k]['text'][max(0,fd-25):fd+30]!r}  {arm}: ...{r[k]['text'][max(0,fd-25):fd+30]!r}")
        print(f"  {arm}: {same}/{len(base)} identical to batch")
json.dump({f"{k[0]}/{k[1]}": {f"{a}.{b}": v for (a, b), v in r.items()} for k, r in allres.items()}, open(OUT + "/texts.json", "w"), indent=1)

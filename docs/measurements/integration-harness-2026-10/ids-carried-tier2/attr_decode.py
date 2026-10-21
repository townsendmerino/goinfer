#!/usr/bin/env python3
# G-ids-3 attribution (ids-carried-2026-10-15.md, "G-ids-3: graded"): python3 -I attr_decode.py OUT CONFIG...
# For each on-arm config: per request, the serve log's ids/fallback fields beside every echoed item whose
# emitted copy (same item id, from an earlier response stream) differs from it.
import gzip, json, os, re, sys
OUT = sys.argv[1]
cfgs = sys.argv[2:]
def resp_items(path):
    items = {}
    for line in gzip.open(path, "rt", errors="replace"):
        if not line.startswith("data:"): continue
        try: ev = json.loads(line[5:])
        except ValueError: continue
        if ev.get("type") == "response.output_item.done":
            it = ev["item"]; items[it.get("id")] = it
    return items
def strip(it):
    return {k: v for k, v in it.items() if k not in ("status",)}
for cfg in cfgs:
    d = f"{OUT}/bodies-{cfg}"
    log = [l for l in open(f"{OUT}/logs/serve-{cfg}.log", errors="replace") if l.startswith("request: POST /v1/responses")]
    seqs = sorted({f.split(".")[0] for f in os.listdir(d)})
    emitted = {}
    print(f"== {cfg}: {len(seqs)} bodies, {len(log)} log lines")
    for i, s in enumerate(seqs):
        req = json.load(gzip.open(f"{d}/{s}.req.json.gz"))
        L = log[i] if i < len(log) else ""
        fb = re.search(r"ids=(\S+) ids_fallback=(\S+)", L)
        diffs = []
        for it in req["input"]:
            e = emitted.get(it.get("id"))
            if e is None: continue
            a, b = strip(it), strip(e)
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    diffs.append((it["id"][-12:], it["type"], k, repr(b.get(k))[:160], repr(a.get(k))[:160]))
        if "decode" in L or diffs:
            print(f"  #{s} {fb.groups() if fb else '?'} prompt={re.search(r'prompt_tokens=(\d+)',L).group(1) if L else '?'}")
            for x in diffs: print("     ", x)
        if os.path.exists(f"{d}/{s}.resp.gz"):
            emitted.update(resp_items(f"{d}/{s}.resp.gz"))

import json, sys
from collections import defaultdict
T = "/home/francis/mycode/goinfer/testdata/decisions/"
items = {json.loads(l)["id"]: json.loads(l) for l in open(T + "items.jsonl")}
ref = {json.loads(l)["id"]: json.loads(l) for l in open(T + "route_a_b0.jsonl")}
gi = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
S = defaultdict(lambda: defaultdict(float))
a = lambda p: max(range(len(p)), key=lambda i: p[i])
for k, it in items.items():
    r, g = ref[k], gi[k]
    kind, tgt = it["kind"], it["target"]
    pr, pg = r["p"], g["distribution"]
    for kk in (kind, "all"):
        s = S[kk]
        s["n"] += 1
        s["agree"] += a(pr) == a(pg)
        s["ref"] += a(pr) == a(tgt)
        s["gi"] += a(pg) == a(tgt)
        s["tv"] += 0.5 * sum(abs(x - y) for x, y in zip(pr, pg))
for kind, s in S.items():
    n = s["n"]
    print(f"{kind:7} n={int(n):3}  argmax agree {s['agree']/n:.3f}  top-1 vs gold: transformers f32 {s['ref']/n:.3f}, goinfer {s['gi']/n:.3f}  mean TV distance {s['tv']/n:.3f}")

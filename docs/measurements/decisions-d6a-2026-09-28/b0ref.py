import json
from collections import defaultdict
T = "/home/francis/mycode/goinfer/testdata/decisions/"
items = {json.loads(l)["id"]: json.loads(l) for l in open(T + "items.jsonl")}
a = lambda p: max(range(len(p)), key=lambda i: p[i])
PROP = {"noul": 9767, "choice": 3219, "score": 72}
def score(path, key="p"):
    S = defaultdict(lambda: [0, 0])
    for l in open(path):
        r = json.loads(l); it = items[r["id"]]
        S[it["kind"]][0] += 1; S[it["kind"]][1] += a(r[key]) == a(it["target"])
    per = {k: v[1] / v[0] for k, v in S.items()}
    w = sum(per[k] * PROP[k] for k in PROP) / sum(PROP.values())
    return per, w
for name, path, key in (("transformers f32, Route A bare-v1 (B0)", T + "route_a_b0.jsonl", "p"),
                        ("transformers f32, JEV-9B trained head", T + "jev9b_ref_f32.jsonl", "p"),
                        ("goinfer Q4_K_M int4, Route A bare-v1", "/home/francis/goinfer-bench/decisions-d6a/b0cmp-goinfer-bare.jsonl", "distribution")):
    per, w = score(path, key)
    print(f"{name:44} top-1 noul {per['noul']:.3f} choice {per['choice']:.3f} score {per['score']:.3f}  | OOD-weighted {w:.3f}")

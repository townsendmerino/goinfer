#!/usr/bin/env python3
"""The Q1 split's readout (README.md, "Q1 split"): C against P per step, and every arm at step 22 against the rule."""
import json, math, os, struct, sys
d = sys.argv[1]
forced = json.load(open(os.path.join(d, "forced.json")))
steps = forced["name_step"] + 1
lp = json.load(open(os.path.join(d, "llama-step22.json")))
tops = lp["completion_probabilities"][0]["top_logprobs"]
ids = {t["token"]: t["id"] for t in tops}
names = [".read", ".list", ".run"]

def rows(arm):
    raw = open(os.path.join(d, arm + "-rows.f32"), "rb").read()
    V = len(raw) // 4 // steps
    flat = struct.unpack("<%df" % (V * steps), raw)
    return [flat[i * V:(i + 1) * V] for i in range(steps)], V

def logsoftmax(r):
    m = max(r)
    z = m + math.log(sum(math.exp(x - m) for x in r))
    return [x - z for x in r]

def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))

arms = {a: rows(a)[0] for a in ("cuda", "cpu") if os.path.exists(os.path.join(d, a + "-rows.f32"))}
if len(arms) == 2:
    print("cosine C vs P per step:", " ".join("%d:%.5f" % (i, cos(arms["cuda"][i], arms["cpu"][i])) for i in range(steps)))
res = {}
for a, rs in arms.items():
    ls = logsoftmax(rs[steps - 1])
    top = sorted(range(len(ls)), key=lambda k: -ls[k])[:5]
    res[a] = {n: ls[ids[n]] for n in names}
    print(f"{a}: " + ", ".join(f"{n} {ls[ids[n]]:.3f}" for n in names) + f"; margin .read-.run {ls[ids['.read']] - ls[ids['.run']]:.3f}; top-5 ids {top}")
res["llama"] = {t["token"]: t["logprob"] for t in tops if t["token"] in names}
print("llama: " + ", ".join(f"{n} {res['llama'].get(n, float('nan')):.3f}" for n in names) + f"; margin {res['llama']['.read'] - res['llama']['.run']:.3f}")
if "cuda" in res and "cpu" in res:
    mc = res["cuda"][".read"] - res["cuda"][".run"]
    mp = res["cpu"][".read"] - res["cpu"][".run"]
    print(f"kernel split: margin C {mc:.3f}, P {mp:.3f}, |C-P| {abs(mc - mp):.3f} nats")

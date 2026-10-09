#!/usr/bin/env python3
"""Step (b') graded as registered (docs/tasks/task-multimodal-support-2026-10.md, "Gemma 4 31B, step (b')"). For each model (the 31B and the plain-bf16 E4B sibling) and each arm (int8int8, int4):
agreement = the fraction of positions where the arm's argmax equals Hugging Face float32's, KL(HF || arm), with a cluster bootstrap over the 12 prompts (10,000 resamples, seed 20261009, 95%).
PASS for an arm: the 31B's agreement is at least the E4B's same-arm agreement minus 5.0 points; PARKED between -5.0 and -10.0; FAIL worse than -10.0 or any non-finite logit.
Usage: g31b_grade.py <31B prefix> <E4B prefix> [hf prefix31 hf prefix e4b]   where <prefix>.seqs.json, <prefix>.<arm>.f32 (goinfer) and <hfprefix>.f32/.json (HF) exist (hf prefixes default to <prefix>.hf)."""
import json, sys
import numpy as np
big, sib = sys.argv[1], sys.argv[2]
hfp = {big: (sys.argv[3] if len(sys.argv) > 3 else big + ".hf"), sib: (sys.argv[4] if len(sys.argv) > 4 else sib + ".hf")}
rng = np.random.default_rng(20261009)
def load(prefix):
    seqs = json.load(open(prefix + ".seqs.json")); lens = [len(s["path"]) for s in seqs]
    meta = json.load(open(hfp[prefix] + ".json")); V = meta["vocab"]; N = sum(lens)
    assert meta["positions"] == lens, (meta["positions"], lens)
    hf = np.fromfile(hfp[prefix] + ".f32", dtype="<f4").reshape(N, V)
    arms = {a: np.fromfile(f"{prefix}.{a}.f32", dtype="<f4").reshape(N, V) for a in ("int8int8", "int4")}
    return lens, hf, arms
def logsm(x): x = x.astype(np.float64); m = x.max(-1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
def per_pos(hf, arm):
    agree = (hf.argmax(-1) == arm.argmax(-1)).astype(float)
    lh, la = logsm(hf), logsm(arm); kl = (np.exp(lh) * (lh - la)).sum(-1)
    return agree, kl, bool(np.isfinite(arm).all())
def boot(vals, lens):
    idx = np.cumsum([0] + lens); clusters = [vals[idx[i]:idx[i + 1]] for i in range(len(lens))]
    pt = float(np.concatenate(clusters).mean())
    bs = [float(np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))]).mean()) for _ in range(10000)]
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
res = {}
for name, prefix in (("31B", big), ("E4B", sib)):
    lens, hf, arms = load(prefix); res[name] = {}
    print(f"{name}: {sum(lens)} positions over {len(lens)} prompts; HF mean top-1 probability {float(np.exp(logsm(hf)).max(-1).mean()):.3f}; HF argmax equals the path token at {float((hf.argmax(-1) == np.concatenate([s['path'] for s in json.load(open(prefix + '.seqs.json'))])).mean()):.3f}")
    for a, lg in arms.items():
        ag, kl, fin = per_pos(hf, lg); p = boot(ag, lens); k = boot(kl, lens)
        res[name][a] = dict(agree=p, kl=k, finite=fin)
        print(f"  {a:9s} agreement {100 * p[0]:6.2f}% [{100 * p[1]:.1f}, {100 * p[2]:.1f}]   mean KL(HF||arm) {k[0]:.4f} [{k[1]:.4f}, {k[2]:.4f}]   finite: {fin}")
print()
verdicts = {}
for a in ("int8int8", "int4"):
    gap = 100 * (res["31B"][a]["agree"][0] - res["E4B"][a]["agree"][0])
    fin = res["31B"][a]["finite"] and res["E4B"][a]["finite"]
    v = "FAIL (non-finite logits)" if not fin else ("PASS" if gap >= -5.0 else ("PARKED" if gap >= -10.0 else "FAIL"))
    verdicts[a] = v
    print(f"{a:9s}: 31B agreement minus E4B's = {gap:+.2f} points (bar: PASS at >= -5.0, PARKED to -10.0) -> {v}")
print("\nROW: " + ("both arms PASS: the 31B row may change" if all(v == "PASS" for v in verdicts.values()) else ("int8int8 only passes" if verdicts["int8int8"] == "PASS" else "not both PASS: the numbers go to the owner")))
json.dump({"results": res, "verdicts": verdicts}, open(big + ".grade.json", "w"), indent=1, default=str)

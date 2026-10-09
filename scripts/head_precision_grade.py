#!/usr/bin/env python3
"""Option D of the --embed-int4 decision, graded as registered (docs/tasks/task-multimodal-support-2026-10.md, "option D").
Per model: d_agree = agree(int8 head) - agree(int4 head) in points, dKL = mean KL(int4 head) - mean KL(int8 head) in nats; paired by position, cluster bootstrap over prompts (32)
(10,000 resamples, seed 20261008, 95% interval). OK: d_agree upper <= 2.3 AND dKL upper <= 0.02. COSTLY: d_agree point >= 4.3 OR dKL point >= 0.05. Else MIXED.
A control result (both arms the int8 pin) must read exactly 0 / 0.   Usage: head_precision_grade.py <dump root>"""
import json, os, sys
import numpy as np
root = sys.argv[1]
rng = np.random.default_rng(20261008)
rows, bad = [], []
for name in sorted(os.listdir(root)):
    f = f"{root}/{name}/result.json"
    if not os.path.exists(f): continue
    r = json.load(open(f))
    if r.get("Error") or not r.get("Int8Head") or not r.get("Int4Head") or not r["Int4Head"].get("KL"):
        bad.append((name, r.get("Error") or "no result")); continue
    a8, a4 = r["Int8Head"], r["Int4Head"]
    kl8, kl4 = np.array(a8["KL"]), np.array(a4["KL"]); ag8, ag4 = np.array(a8["Agree"], float), np.array(a4["Agree"], float)
    n = kl8.shape[0]
    def stats(idx):
        return 100 * (ag8[idx].mean() - ag4[idx].mean()), kl4[idx].mean() - kl8[idx].mean()
    d_agree, d_kl = stats(np.arange(n))
    bs = np.array([stats(rng.integers(0, n, n)) for _ in range(10000)])
    lo_a, hi_a = np.percentile(bs[:, 0], [2.5, 97.5]); lo_k, hi_k = np.percentile(bs[:, 1], [2.5, 97.5])
    if r.get("Control"):
        ok = d_agree == 0 and d_kl == 0
        print(f"CONTROL {name}: d_agree {d_agree:+.4f} dKL {d_kl:+.6f} -> {'EXACTLY ZERO (the statistic is noiseless)' if ok else 'NOT ZERO: the statistic has noise of its own'}"); continue
    verdict = "OK" if (hi_a <= 2.3 and hi_k <= 0.02) else ("COSTLY" if (d_agree >= 4.3 or d_kl >= 0.05) else "MIXED")
    rows.append((name, verdict))
    print(f"{name:24s} agree int8 {100 * ag8.mean():5.1f}% int4 {100 * ag4.mean():5.1f}% | d_agree {d_agree:+5.2f} pts [{lo_a:+5.2f}, {hi_a:+5.2f}] | KL int8 {kl8.mean():.4f} int4 {kl4.mean():.4f} "
          f"dKL {d_kl:+.4f} [{lo_k:+.4f}, {hi_k:+.4f}] ratio {kl4.mean() / kl8.mean():.2f}x | {verdict} | head {a8['HeadTable']}/{a4['HeadTable']} {a8['SecPerStep']:.2f}/{a4['SecPerStep']:.2f} s/step")
for name, why in bad: print(f"{name:24s} NOT RUN: {why}")
tied = {"qwen2.5-0.5b-instruct", "qwen3-1.7b-bf16", "qwen25vl-3b-instruct", "gemma-3-4b-it"}
v = dict(rows)
nt = sum(1 for n in tied if v.get(n) == "COSTLY"); nu = sum(1 for n in v if n not in tied and v[n] == "COSTLY")
if rows:
    print(f"\nreadings: {len(rows)} of 7 models graded, {len(bad)} not run; tied-head COSTLY {nt}/4, untied COSTLY {nu}")
    if len(rows) == 7 and all(x == "OK" for x in v.values()): print("MAP: all seven OK -> option C is justified on quality (the Mac work is the owner's call)")
    elif nt >= 2 and nu == 0: print("MAP: the tied-head models are costly and the untied are not -> candidate rule: default off for tied-head models, resolved per model at load")
    elif nt >= 2 and nu >= 1: print("MAP: both kinds costly -> option B (off everywhere), speed given back")
    else: print("MAP: none of the registered patterns -> the numbers go to the owner as they are")

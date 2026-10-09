#!/usr/bin/env python3
"""The 896-pixel investigation's A/A table: two anchor summaries (scripts/anchor_s10h_hf.py's anchor-summary.json) from dumps that differ ONLY in the CPU ("off") arm's attention variant
(GOINFER_S10H_KNOBS), against the same Hugging Face float32 logits and the same unchanged resident ("on") arm. It prints, per image, the step-0 cosine to HF of each CPU variant, the A/A shift
(variant minus baseline: how far one correct CPU implementation moves by itself), the registered rule's delta (on minus off) against each, and the 36-pair means.
Usage: python3 scripts/s896_aa_table.py <baseline anchor-summary.json> <variant anchor-summary.json>"""
import json, sys
import numpy as np
a, b = (json.load(open(p)) for p in sys.argv[1:3])
names = {"gemma3_preprocess_image.png": "896² (803 rows)", "qwen25vl_preprocess_image.png": "4x6 (89)", "formula.png": "formula (1034)", "table.png": "table (991)"}
print("per image, step 0: cos(arm,HF); delta = cos(on) - cos(off). base = the baseline CPU arm, var = the variant, on = the resident prefill (unchanged)")
print(f"{'image':16s} {'cos base':>9s} {'cos var':>8s} {'A/A var-base':>13s} {'cos on':>7s} | {'delta on-base':>14s} {'delta on-var':>13s} | 9-step mean delta: {'on-base':>8s} {'on-var':>8s} | KL base  var   on")
tot = [[], []]
for k, nm in names.items():
    x, y = a[k], b[k]
    d1, d2 = np.array(x["delta"]), np.array(y["delta"]); tot[0] += list(d1); tot[1] += list(d2)
    print(f"{nm:16s} {x['cos_off'][0]:9.4f} {y['cos_off'][0]:8.4f} {y['cos_off'][0] - x['cos_off'][0]:+13.4f} {x['cos_on'][0]:7.4f} | {x['step0_delta']:+14.4f} {y['step0_delta']:+13.4f} |                   {d1.mean():+8.4f} {d2.mean():+8.4f} | {x['kl_off']:.3f} {y['kl_off']:.3f} {x['kl_on']:.3f}")
aa = np.concatenate([np.array(b[k]["cos_off"]) - np.array(a[k]["cos_off"]) for k in names])
print(f"\nA/A (var - base) over all {len(aa)} (image, step) pairs: mean {aa.mean():+.4f}, sd {aa.std(ddof=1):.4f}, |max| {np.abs(aa).max():.4f}")
print(f"registered rule's mean delta (FAIL below -0.005, PASS from -0.002): on-base {np.mean(tot[0]):+.4f}, on-var {np.mean(tot[1]):+.4f}; swing from the CPU attention variant alone {np.mean(tot[1]) - np.mean(tot[0]):+.4f}")
print(f"registered step-0 deltas on-base {[round(a[k]['step0_delta'], 4) for k in names]}, on-var {[round(b[k]['step0_delta'], 4) for k in names]}")

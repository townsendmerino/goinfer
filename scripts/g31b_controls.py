#!/usr/bin/env python3
"""The plumbing controls for step (b') (docs/tasks/task-multimodal-support-2026-10.md, "Gemma 4 31B, step (b')", controls 1-3), run on a 31B-shaped tiny BEFORE any 31B reading:
  1. streaming against ordinary on testdata/gemma4-dense-twogeom-tiny (dense, two attention geometries, K=V on the global layer, softcap, tied head, sliding window 4) over sequences LONGER than the
     window so the window engages: per-position logits cosine >= 0.9999999 and max |difference| <= 1e-5 at every position;
  2. each registered planted defect, alone, must break that bar (nosoftcap, nonorm, swap, fullmask, kvtie, noscale);
  3. determinism: the streaming script run twice gives bit-identical logits.
Exit status 0 only if all three hold. Usage: ~/g4venv/bin/python -I scripts/g31b_controls.py [model dir]"""
import importlib.util, os, sys
import numpy as np
spec = importlib.util.spec_from_file_location("g31b", os.path.join(os.path.dirname(os.path.abspath(__file__)), "g31b_hf_stream.py")); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata", "gemma4-dense-twogeom-tiny")
rng = np.random.default_rng(31)
V = g.text_config(d).vocab_size
seqs = [{"ids": [int(x) for x in rng.integers(1, V, n)], "path": [int(x) for x in rng.integers(1, V, k)]} for n, k in ((11, 6), (14, 5), (9, 7))]
win = g.text_config(d).sliding_window
assert all(len(s["ids"]) + len(s["path"]) - 1 > win for s in seqs), "every sequence must be longer than the sliding window"
def cmp(a, b):
    cs = [float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))) for xs, ys in zip(a, b) for x, y in zip(xs.astype(np.float64), ys.astype(np.float64))]
    return min(cs), max(float(np.abs(xs - ys).max()) for xs, ys in zip(a, b))
ref = g.ordinary(d, seqs)
ok = True
base = g.stream(d, seqs, "")
cmin, dmax = cmp(base, ref); c1 = cmin >= 0.9999999 and dmax <= 1e-5; ok &= c1
print(f"control 1: streaming against ordinary (window {win}, sequences of {[len(s['ids']) + len(s['path']) - 1 for s in seqs]} tokens): min cosine {cmin:.10f}, max |diff| {dmax:.3g} -> {'HELD' if c1 else 'NOT HELD'} (bars 0.9999999 and 1e-5)")
for defect in ("nosoftcap", "nonorm", "swap", "fullmask", "kvtie", "noscale"):
    out = g.stream(d, seqs, defect); cmin, dmax = cmp(out, ref); red = not (cmin >= 0.9999999 and dmax <= 1e-5); ok &= red
    print(f"control 2: planted defect {defect:10s}: min cosine {cmin:.7f}, max |diff| {dmax:.3g} -> {'RED (the bar sees it)' if red else 'NOT RED: the control cannot see this defect'}")
again = g.stream(d, seqs, ""); same = all(np.array_equal(x, y) for x, y in zip(again, base)); ok &= same
print(f"control 3: the streaming script twice: bit-identical logits: {same}")
print("ALL CONTROLS HELD" if ok else "CONTROLS NOT HELD")
sys.exit(0 if ok else 1)

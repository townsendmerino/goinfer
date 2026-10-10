#!/usr/bin/env python3
"""G-S14c4d's grading (docs/tasks/task-multimodal-support-2026-10.md, registered before the code). Usage:
  s14c4d_grade.py <refs.json> <hf.json> <go.json> [<old-go.json>]
go.json holds the arms f32, int4, int4h8 and, from GOINFER_S14C4_PF=1, f32pf, int4pf, int4h8pf (decoder TestQwen3ASRWER_arms); old-go.json is G-S14c4's 2026-10-09 output, the control that the unforced path did not move.
Normalisation, WER, the bootstrap and the PASS / PARKED / FAIL rule are scripts/s14c4_grade.py's (G-S14c4b), applied to the arm int4pf against f32."""
import json, re, sys
from collections import Counter
import numpy as np
refs, hf, go = (json.load(open(p)) for p in sys.argv[1:4]); old = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
ids = sorted(set(refs) & set(go["f32"]))
TITLES = {"mr": "mister", "mrs": "missus", "dr": "doctor"}
def norm(t, hyp=True):
    if hyp:
        t = t.split("<asr_text>", 1)[-1]
        t = re.sub(r"<[^<>]{1,40}>", " ", t)
    t = t.lower().replace("-", " ")
    t = re.sub(r"\b(mr|mrs|dr)\.", lambda m: TITLES[m.group(1)], t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    return t.split()
def dist(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]; d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
    return d[-1]
def errs(h): return {i: (dist(norm(refs[i], False), norm(h.get(i, ""))), len(norm(refs[i], False))) for i in ids}
def wer(e, sel=None):
    sel = sel or ids; return 100.0 * sum(e[i][0] for i in sel) / sum(e[i][1] for i in sel)
ctrl = lambda h: sum(1 for i in ids if re.search(r"<[^<>]{1,40}\|>|<\|[^<>]{1,40}>", h.get(i, "").split("<asr_text>", 1)[0] if "<asr_text>" in h.get(i, "") else h.get(i, "")))
if "f32pf" not in go:
    sys.exit("go.json has no partial-force arms (run with GOINFER_S14C4_PF=1)")
print(f"{len(ids)} clips, {sum(len(norm(refs[i], False)) for i in ids)} reference words")
# control: this binary's unforced path against the 2026-10-09 record
if old:
    for a in ("f32", "int4", "int4h8"):
        if a in old:
            same = sum(1 for i in ids if go[a].get(i) == old[a].get(i))
            print(f"control: arm {a} byte-equal to 2026-10-09's on {same}/{len(ids)} clips")
rows = [("R  transformers float32", hf), ("f32", go["f32"]), ("f32pf", go["f32pf"]), ("int4", go["int4"]), ("int4pf", go["int4pf"]), ("int4h8", go["int4h8"]), ("int4h8pf (record)", go["int4h8pf"])]
E = {n: errs(h) for n, h in rows}
for n, h in rows:
    print(f"  {n:26s} WER {wer(E[n]):6.2f}%   control tokens in {ctrl(h)} replies")
# G-S14c4d-a
F, P = go["f32"], go["f32pf"]
eq = [i for i in ids if F[i].strip() == P[i].strip()]
dW = abs(wer(E["f32pf"]) - wer(E["f32"]))
okA = len(eq) >= 72 * len(ids) / 73 and dW <= 0.30
print(f"\nG-S14c4d-a: f32pf byte-equal to f32 on {len(eq)}/{len(ids)} clips (bar >= 72/73 scaled: {72 * len(ids) / 73:.1f}); |WER diff| = {dW:.2f} pts (bar <= 0.30): {'PASS' if okA else 'FAIL'}")
for i in ids:
    if i not in eq:
        a, b = norm(F[i]), norm(P[i]); k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        print(f"    {i}: first differing word {k}: f32 {a[k:k+3]} f32pf {b[k:k+3]}")
# G-S14c4d-b: G-S14c4b's rule, int4pf against f32
rng = np.random.default_rng(20261008); idx = np.array(ids)
def delta(sel): return wer(E["int4pf"], sel) - wer(E["f32"], sel)
d0 = delta(ids); boot = np.array([delta(list(rng.choice(idx, len(idx)))) for _ in range(10000)]); lo, hi = np.percentile(boot, [2.5, 97.5])
I = go["int4pf"]; nd = sum(1 for i in ids if norm(I[i]) != norm(F[i])); cI = ctrl(I)
verdict = "PASS" if (d0 <= 0.5 and hi <= 1.5) else ("PARKED" if d0 <= 1.5 else "FAIL")
if cI > 3 and verdict == "PASS": verdict = "PARKED (control tokens)"
print(f"\nG-S14c4d-b: WER_int4pf - WER_f32 = {d0:+.2f} pts (95% interval [{lo:+.2f}, {hi:+.2f}]); clips where int4pf's words differ from f32's: {nd}; control tokens in int4pf: {cI} (> 3 parks)")
print(f"  verdict: {verdict}   (the unforced arm read +4.17 [+0.61, +9.52] with 33 control tokens)")
# record only: what each arm writes at the three positions after "assistant\n" (the forced word counts as written)
def cls_first(t):  # for a partial-force arm the forced word is stored in front, so position 1 reads "language" by construction
    if t.startswith("language"): return "language"
    if t.startswith("<|"): return "control token first"
    return "empty" if not t.strip() else "other first"
def cls_name(t):
    m = re.match(r"(?:<\|im_start\|>)?language( ?)([^<\s]*)(<asr_text>)?", t)
    if not m: return "no 'language' prefix"
    if not m.group(2): return "no language name"
    return "name written" if m.group(3) else "name, no <asr_text>"
def cls_word(t):
    if "<asr_text>" not in t: return "no <asr_text>"
    rest = t.split("<asr_text>", 1)[1]
    if not rest.strip(): return "empty transcript"
    if re.match(r"\s*<\|", rest): return "control token first"
    return "word"
print("\nrecord only: first-token classes (position 1: the first generated token, or the forced word; position 2: the name; position 3: the first transcript token)")
for n in ("f32", "f32pf", "int4", "int4pf", "int4h8", "int4h8pf"):
    h = go[n]
    c1, c2, c3 = Counter(cls_first(h[i]) for i in ids), Counter(cls_name(h[i]) for i in ids), Counter(cls_word(h[i]) for i in ids)
    print(f"  {n:9s} p1 {dict(c1)}  p2 {dict(c2)}  p3 {dict(c3)}")
langs = lambda h: {i: (re.match(r"(?:<\|im_start\|>)?language ?([^<\s]*)", h[i]) or [None, ""])[1] for i in ids}
lf, lp = langs(go["f32"]), langs(go["int4pf"])
print(f"  language name: int4pf differs from f32 on {sum(1 for i in ids if lf[i] != lp[i])}/{len(ids)} clips (English data: a design property of keeping detection is not tested here)")

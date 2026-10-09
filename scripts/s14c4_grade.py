#!/usr/bin/env python3
"""G-S14c4's grading (docs/tasks/task-multimodal-support-2026-10.md, registered before the harness). Usage:
  s14c4_grade.py <refs.json> <hf.json> <goinfer.json> [gemma.json]
hf.json is the reference arm R, goinfer.json holds the arms f32 and int4 (decoder TestQwen3ASRWER_arms), gemma.json (optional) the record-only Gemma 4 E4B arm D. The control S is F shifted by one clip.
Normalisation: strip `language X<asr_text>` and <|..|> control tokens from the hypothesis; lowercase; Mr./Mrs./Dr. -> mister/missus/doctor; hyphens to spaces; drop everything except letters, digits and apostrophes."""
import json, re, sys
import numpy as np
refs, hf, go = (json.load(open(p)) for p in sys.argv[1:4]); gemma = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
ids = sorted(set(refs) & set(go["f32"]))
TITLES = {"mr": "mister", "mrs": "missus", "dr": "doctor"}
def norm(t, hyp=True):
    if hyp:
        t = t.split("<asr_text>", 1)[-1]
        t = re.sub(r"<[^<>]{1,40}>", " ", t)  # any special-token form: <|im_start|>, Gemma's <channel|>, <|channel>
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
def errs(hyps):  # per clip (errors, ref words)
    return {i: (dist(norm(refs[i], False), norm(hyps.get(i, ""))), len(norm(refs[i], False))) for i in ids}
def wer(e, sel=None):
    sel = sel or ids; return 100.0 * sum(e[i][0] for i in sel) / sum(e[i][1] for i in sel)
ctrl = lambda h: sum(1 for i in ids if re.search(r"<[^<>]{1,40}\|>|<\|[^<>]{1,40}>", h.get(i, "").split("<asr_text>", 1)[0] if "<asr_text>" in h.get(i, "") else h.get(i, "")))
R, F, I = hf, go["f32"], go["int4"]
S = {ids[k]: F[ids[(k + 1) % len(ids)]] for k in range(len(ids))}
eR, eF, eI, eS = errs(R), errs(F), errs(I), errs(S)
print(f"{len(ids)} clips, {sum(eF[i][1] for i in ids)} reference words")
for name, e, h in (("R  transformers float32", eR, R), ("F  goinfer native weights", eF, F), ("I  goinfer int4 (serve defaults)", eI, I), ("S  control: F on the next clip's audio", eS, S)):
    print(f"  {name:42s} WER {wer(e):6.2f}%   control tokens in {ctrl(h)} replies")
if gemma:
    quoted = lambda t: max(re.findall(r'"([^"]{8,})"', t), key=len) if re.findall(r'"([^"]{8,})"', t) else t
    eD = errs(gemma); eD2 = errs({i: quoted(gemma.get(i, "")) for i in ids})
    print(f"  {'D  Gemma 4 E4B, raw replies (record only)':42s} WER {wer(eD):6.2f}%   ({sum(1 for i in ids if i in gemma)} replies)")
    print(f"  {'D' + chr(39) + ' Gemma 4 E4B, longest quoted span':42s} WER {wer(eD2):6.2f}%   (its replies carry a preamble and thinking-channel markers; the quoted span is a best-effort extraction)")
# G-S14c4a
eq = [i for i in ids if R[i].replace("<|im_end|>", "").strip() == F[i].strip()]
print(f"\nG-S14c4a: F byte-equal to R on {len(eq)}/{len(ids)} clips (bar >= 70/73 scaled: {0.95 * len(ids):.1f}); |WER_F - WER_R| = {abs(wer(eF) - wer(eR)):.2f} pts (bar <= 0.30)")
for i in ids:
    if i not in eq:
        a, b = norm(R[i]), norm(F[i]); k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        print(f"    {i}: first differing word {k}: R {a[k:k+3]} F {b[k:k+3]}")
okA = len(eq) >= 0.95 * len(ids) and abs(wer(eF) - wer(eR)) <= 0.3
# G-S14c4b
rng = np.random.default_rng(20261008); idx = np.array(ids)
def delta(sel): return wer(eI, sel) - wer(eF, sel)
d0 = delta(ids); boot = np.array([delta(list(rng.choice(idx, len(idx)))) for _ in range(10000)]); lo, hi = np.percentile(boot, [2.5, 97.5])
nd = sum(1 for i in ids if norm(I[i]) != norm(F[i])); cI = ctrl(I)
print(f"\nG-S14c4b: WER_I - WER_F = {d0:+.2f} pts (95% interval [{lo:+.2f}, {hi:+.2f}]); clips where I's words differ from F's: {nd}; control tokens in I: {cI} (> 3 parks)")
verdict = "PASS" if (d0 <= 0.5 and hi <= 1.5) else ("PARKED" if d0 <= 1.5 else "FAIL")
if cI > 3 and verdict == "PASS": verdict = "PARKED (control tokens)"
print(f"  verdict: {verdict}")
print(f"\nControls: F against F = {wer(eF) - wer(eF):+.2f} pts (must be 0); S WER {wer(eS):.1f}% (must be > 80)")
print("G-S14c4a:", "PASS" if okA else "NOT MET", "| G-S14c4b:", verdict, "| controls:", "ok" if wer(eS) > 80 else "S TOO LOW")

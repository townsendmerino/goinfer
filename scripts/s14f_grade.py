#!/usr/bin/env python3
"""G-S14f3's grading (docs/tasks/task-multimodal-support-2026-10.md, "G-S14f", registered before the code). Usage: s14f_grade.py <refs.json> <hf.json> <goinfer.json>
Both arms hold {id: {text, lang}}. The normaliser and the WER are G-S14c4's. PASS: texts byte-equal on at least 70 of 73 clips (scaled to the clip count) and |WER_goinfer - WER_transformers| <= 0.30 points.
Reported, not graded: the language each side detected."""
import json, re, sys
refs, hf, go = (json.load(open(p)) for p in sys.argv[1:4])
ids = sorted(set(refs) & set(hf) & set(go))
TITLES = {"mr": "mister", "mrs": "missus", "dr": "doctor"}
def norm(t):
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
def wer(arm): return 100.0 * sum(dist(norm(refs[i]), norm(arm[i]["text"])) for i in ids) / sum(len(norm(refs[i])) for i in ids)
eq = [i for i in ids if hf[i]["text"] == go[i]["text"]]
wH, wG = wer(hf), wer(go)
print(f"{len(ids)} clips, {sum(len(norm(refs[i])) for i in ids)} reference words")
print(f"  transformers float32 WER {wH:.2f}%   goinfer float32 WER {wG:.2f}%")
need = 70 * len(ids) / 73
ok = len(eq) >= need and abs(wG - wH) <= 0.30
print(f"G-S14f3: byte-equal texts on {len(eq)}/{len(ids)} (bar >= {need:.1f}); |WER diff| = {abs(wG - wH):.2f} pts (bar <= 0.30): {'PASS' if ok else 'FAIL'}")
for i in ids:
    if i not in eq:
        a, b = norm(hf[i]["text"]), norm(go[i]["text"]); k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        print(f"    {i}: first differing word {k}: transformers {a[k:k+3]} goinfer {b[k:k+3]}")
dl = [i for i in ids if hf[i]["lang"] != go[i]["lang"]]
print(f"record: language detected differently on {len(dl)}/{len(ids)} clips" + ("" if not dl else f": {[(i, hf[i]['lang'], go[i]['lang']) for i in dl]}"))

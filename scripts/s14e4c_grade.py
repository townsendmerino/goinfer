#!/usr/bin/env python3
"""G-S14e4c's record (docs/tasks/task-multimodal-support-2026-10.md, registered before the code; nothing here is graded). Usage:
  s14e4c_grade.py <ref-dir> <out-dir>
ref-dir is scripts/pin_voxtral_real.py's output with --language none (libri.gen.json, libri6x.gen.json and the HF log's decoded text in ref.log); out-dir holds reply-{f32,int4}-{libri,libri6x}.json from scripts/s14e4_request.py.
Prints, per clip: the float32 served reply against the reference (byte-equal or the first differing word), and the int4 reply against float32 (word-level edit distance, the first characters, control tokens)."""
import json, os, re, sys
ref, out = sys.argv[1:3]
log = open(os.path.join(ref, "ref.log"), errors="replace").read().replace("\r", "\n")
HF = {m.group(1): m.group(3) for m in re.finditer(r"^(libri6x|libri): \d+ prompt ids.*?generated (\d+) tokens in \d+s: '(.*)'$", log, re.M)}
def words(t): return re.sub(r"[^a-z0-9' ]", " ", t.lower().replace("-", " ")).split()
def dist(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]; d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
    return d[-1]
for clip in ("libri", "libri6x"):
    try:
        f32 = json.load(open(f"{out}/reply-f32-{clip}.json")); i4 = json.load(open(f"{out}/reply-int4-{clip}.json"))
    except OSError as e:
        print(f"{clip}: missing a reply ({e})"); continue
    gen = json.load(open(f"{ref}/{clip}.gen.json")) if os.path.exists(f"{ref}/{clip}.gen.json") else []
    h = HF.get(clip)
    print(f"{clip}: float32 served: {f32['usage']['completion_tokens']} completion tokens (reference generated {len(gen)} incl. the stop), finish {f32['finish']}, {f32['seconds']:.0f}s")
    if h is None: print("  no reference text found in ref.log")
    elif f32["reply"] == h: print("  float32 served reply BYTE-EQUAL to the reference")
    else:
        a, b = words(h), words(f32["reply"]); k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        print(f"  float32 served reply DIFFERS from the reference at word {k}: ref {a[k:k+4]} served {b[k:k+4]}")
    a, b = words(f32["reply"]), words(i4["reply"])
    first = i4["reply"][:40]
    ctl = re.findall(r"<\|[^<>]{1,40}\|>|\[[A-Z_/]{2,20}\]", i4["reply"])
    print(f"  int4 served ({i4['seconds']:.0f}s, {i4['usage']['completion_tokens']} tokens, finish {i4['finish']}): {dist(a, b)} word edits against float32 over {len(a)} words; control tokens {ctl}; starts {first!r}")

#!/usr/bin/env python3
"""G-S19c's served grade (docs/tasks/task-multimodal-support-2026-10.md, "S19"): each image's reply on the new arm against the
control arm (the same binary with GOINFER_BATCHED_PREFILL=0: today's CPU prefill and upload). PASS per image: identical, or the
first differing token is a near-tie under the control's own top-3 (R10: p(new's token) >= half p(control's top token)).
Usage: grade_s19_served.py <g3 out dir> [new label] [control label]"""
import glob, json, math, os, sys
out = sys.argv[1]
new, ctl = (sys.argv[2:4] + ["new", "ctl"])[:2] if len(sys.argv) > 2 else ("new", "ctl")
ok = True
for f in sorted(glob.glob(f"{out}/{ctl}-*.json")):
    name = os.path.basename(f)[len(ctl) + 1:-5]
    b = json.load(open(f))["logprobs"]
    a = json.load(open(f"{out}/{new}-{name}.json"))["logprobs"]
    k = next((i for i in range(min(len(a), len(b))) if a[i]["token"] != b[i]["token"]), None)
    if k is None:
        print(f"{name}: {len(a)} tokens IDENTICAL" if len(a) == len(b) else f"{name}: lengths differ ({len(a)} vs {len(b)})")
        ok = ok and len(a) == len(b)
        continue
    top = b[k]["top_logprobs"]
    ptop = math.exp(top[0]["logprob"])
    pnew = next((math.exp(t["logprob"]) for t in top if t["token"] == a[k]["token"]), 0.0)
    tie = pnew >= ptop / 2
    ctx = "".join(t["token"] for t in b[max(0, k - 8):k])
    print(f"{name}: first difference at generated token {k} after {ctx!r}: control {b[k]['token']!r} p={ptop:.3f}, new {a[k]['token']!r} (control's p={pnew:.3f}); near-tie: {tie}")
    ok = ok and tie
print("G-S19c served:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

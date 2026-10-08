#!/usr/bin/env python3
"""G-S16c's served verdict for one model (run-s16-night.sh step 2): today's path (--exact-prefill) against S16, the Metal
arms' replies and log-probabilities. PASS: identical replies, or the first differing token at a near-tie in today's
log-probabilities (p(other) >= half p(top)). Usage: served_verdict.py <out dir> <model dir name>."""
import json
import math
import sys

out, fam = sys.argv[1:3]


def load(d):
    return (open(f"{out}/{d}-{fam}/gs3c-reply-{fam}-metal.txt").read(),
            json.load(open(f"{out}/{d}-{fam}/gs3c-logprobs-{fam}-metal.json")))


try:
    (ro, lo), (rn, ln) = load("2-old"), load("2-new")
except Exception as e:  # a missing arm voids the step
    print(f"G-S16c served {fam}: VOID ({e})")
    sys.exit(1)
if ro == rn:
    print(f"G-S16c served {fam}: PASS, IDENTICAL replies (today's path against S16)")
    sys.exit(0)
i = 0
while i < min(len(lo), len(ln)) and lo[i]["token"] == ln[i]["token"]:
    i += 1
if i >= min(len(lo), len(ln)):
    print(f"G-S16c served {fam}: FAIL, the replies differ in length only")
    sys.exit(1)
top = lo[i]["top_logprobs"]
ptop = math.exp(top[0]["logprob"])
po = [math.exp(c["logprob"]) for c in top if c["token"] == ln[i]["token"]]
near = bool(po and po[0] >= ptop / 2)
print(f"{fam}: first differing generated token {i}: today's {lo[i]['token']!r}, S16 {ln[i]['token']!r}; today's top-3: "
      + ", ".join(f"{c['token']!r} {math.exp(c['logprob']):.3f}" for c in top))
print(f"G-S16c served {fam}: {'PASS' if near else 'FAIL'} (near-tie: p(other) >= {ptop / 2:.3f}: {near})")
sys.exit(0 if near else 1)

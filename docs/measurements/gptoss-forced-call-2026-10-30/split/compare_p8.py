#!/usr/bin/env python3
"""P8 and its follow-ons, graded by the rule in README.md, "P8, registered". Reads <arm>-rows.f32 for each CPU arm present
in <dir> and llama-step22.json (L, this morning's reading).
Usage: compare_p8.py <dir>. Prints one line per arm and a VERDICT line for P8 and, when present, for the follow-ons.
Exits 0 whatever the verdict; exits 2 when P8's rows are missing."""
import json
import os
import sys

import numpy as np

d = sys.argv[1]
forced = json.load(open(os.path.join(d, "forced.json")))
steps = forced["name_step"] + 1
tops = json.load(open(os.path.join(d, "llama-step22.json")))["completion_probabilities"][0]["top_logprobs"]
ids = {t["token"]: t["id"] for t in tops}
names = [".read", ".list", ".search", ".run"]
L = {t["token"]: t["logprob"] for t in tops}
mL = L[".read"] - L[".run"]


def step22(arm):
    p = os.path.join(d, arm + "-rows.f32")
    if not os.path.exists(p):
        return None
    r = np.fromfile(p, dtype="<f4").astype(np.float64)
    r = r.reshape(steps, -1)[steps - 1]
    r = r - r.max()
    ls = r - np.log(np.exp(r).sum())
    top = [int(i) for i in np.argsort(-ls)[:5]]
    m = ls[ids[".read"]] - ls[ids[".run"]]
    print(f"{arm}: " + ", ".join(f"{n} {ls[ids[n]]:.3f}" for n in names) + f"; margin {m:.3f}; top-5 ids {top}")
    return m


print("llama: " + ", ".join(f"{n} {L[n]:.3f}" for n in names) + f"; margin {mL:.3f}")
mP = step22("cpu")
m8 = step22("cpu-int8")
if m8 is None:
    print("VERDICT P8: no rows (cpu-int8-rows.f32 missing)")
    sys.exit(2)
if m8 > mL + 2:
    v8 = "OVER: more than 2 nats over L's; no branch covers it; follow-ons do not run"
elif m8 > mL - 2:
    v8 = "QUANTIZATION: within 2 nats of L's; P88 and P4g run"
else:
    v8 = "CODE: 2 or more nats under L's; follow-ons do not run"
print(f"VERDICT P8: margin {m8:.3f} vs L {mL:.3f} (diff {m8 - mL:+.3f}); {v8}")
held = {}
for arm, tag in (("cpu-int8int8", "P88"), ("cpu-int4g32", "P4g")):
    m = step22(arm)
    if m is None:
        continue
    held[tag] = m > m8 - 2  # the registration's two branches: within 2 nats of P8, or 2 or more under it
    print(f"VERDICT {tag}: margin {m:.3f} vs P8 {m8:.3f} (diff {m - m8:+.3f}); {'holds' if held[tag] else '2 or more nats under P8'}")
if len(held) == 2:
    combo = {
        (True, False): "the int4 re-rounding of the weights",
        (False, True): "the per-row activation scale",
        (False, False): "both the int4 weights and the per-row activations",
        (True, True): "only the int4-plus-per-row combination (reported)",
    }[(held["P88"], held["P4g"])]
    print(f"VERDICT follow-ons: P88 {'holds' if held['P88'] else 'fails'}, P4g {'holds' if held['P4g'] else 'fails'}: the share is {combo}")

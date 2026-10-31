#!/usr/bin/env python3
"""The int4 quantizer arms, graded by README.md, "P8 addendum: the int4 quantizers". Each scheme arm's step-22 margin
(.read - .run) is read against P4g (the default max/7 int4 rounding, per-32 activations) and P8 (int8 weights, f32
activations), both already measured. Usage: compare_q4s.py <dir>. Exits 2 when P4g's or P8's rows are missing."""
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


def lsm(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def load(arm):
    p = os.path.join(d, arm + "-rows.f32")
    if not os.path.exists(p):
        return None
    return lsm(np.fromfile(p, dtype="<f4").astype(np.float64).reshape(steps, -1))


def margin(ls, arm):
    s = ls[steps - 1]
    m = s[ids[".read"]] - s[ids[".run"]]
    top = [int(i) for i in np.argsort(-s)[:5]]
    print(f"{arm}: " + ", ".join(f"{n} {s[ids[n]]:.3f}" for n in names) + f"; margin {m:.3f}; top-5 ids {top}")
    return m


p8, p4g = load("cpu-int8"), load("cpu-int4g32")
if p8 is None or p4g is None:
    print("VERDICT: P8's or P4g's rows are missing")
    sys.exit(2)
m8, m4 = margin(p8, "cpu-int8 (P8)"), margin(p4g, "cpu-int4g32 (P4g)")
pr = np.exp(p8)
for scheme in ("fullrange", "mse"):
    arm = "cpu-int4g32-" + scheme
    ls = load(arm)
    if ls is None:
        print(f"VERDICT {scheme}: no rows")
        continue
    m = margin(ls, arm)
    if m > m8 - 2:
        v = "RECOVERS: within 2 nats of P8"
    elif m >= m4 + 2:
        v = "PARTIAL: 2 or more nats over P4g, not within 2 of P8"
    elif m > m4 - 2:
        v = "NO CHANGE: within 2 nats of P4g"
    else:
        v = "WORSE: 2 or more nats under P4g"
    kl = float((pr * (p8 - ls)).sum())
    print(f"VERDICT {scheme}: margin {m:.3f} (P4g {m4:.3f}, P8 {m8:.3f}); {v}; KL(P8||arm) summed over the steps {kl:.3f}")

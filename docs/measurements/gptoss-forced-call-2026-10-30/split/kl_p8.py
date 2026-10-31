#!/usr/bin/env python3
"""P8's descriptive companion (README.md, "P8, read"): for every goinfer arm present in <dir>, the KL from P8 (the int8,
f32-activation CPU arm) at each of the 23 steps, its sum, and the steps where the arm's argmax is not the forced token.
Not graded by the registration; it shows whether step 22 is typical. Usage: kl_p8.py <dir>"""
import json
import os
import sys

import numpy as np

d = sys.argv[1]
forced = json.load(open(os.path.join(d, "forced.json")))
steps = forced["name_step"] + 1


def lsm(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def rows(arm):
    return lsm(np.fromfile(os.path.join(d, arm + "-rows.f32"), dtype="<f4").astype(np.float64).reshape(steps, -1))


ref = rows("cpu-int8")
p = np.exp(ref)
for arm in ("cpu-int8", "cpu-int8int8", "cpu-int4g32", "cpu", "cuda"):
    if not os.path.exists(os.path.join(d, arm + "-rows.f32")):
        continue
    a = rows(arm)
    kl = (p * (ref - a)).sum(axis=1)
    miss = [s for s in range(len(forced["forced"])) if int(np.argmax(a[s])) != forced["forced"][s]]
    print(f"{arm:13s} KL(P8||arm) sum {kl.sum():.3f}, step 22 {kl[22]:.3f}, max over 0-21 {kl[:22].max():.3f}; "
          f"argmax off the forced token at steps {miss}")

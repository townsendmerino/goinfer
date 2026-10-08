#!/usr/bin/env python3
"""G-S5e option (b)'s filter check (docs/tasks/task-multimodal-support-2026-10.md): scipy.signal.resample_poly (defaults:
Kaiser beta 5.0, zero padding) of a deterministic signal from 44.1 kHz and from 48 kHz to 16 kHz, written to
testdata/resample_poly_golden.json for multimodal's TestResample_matchesResamplePoly.
Run from the repo root: python3 scripts/pin_resample_poly_golden.py"""
import json
import math

import numpy as np
import scipy
from scipy.signal import resample_poly


def signal(n, rate):
    t = np.arange(n) / rate
    rng = np.random.default_rng(5)
    return (0.4 * np.sin(2 * math.pi * 440 * t) + 0.2 * np.sin(2 * math.pi * 7900 * t)
            + 0.1 * np.sin(2 * math.pi * 12000 * t) + 0.05 * rng.standard_normal(n)).astype(np.float32)


out = {"scipy": scipy.__version__, "cases": []}
for rate, n in [(44100, 3001), (48000, 3001), (22050, 1001), (8000, 777)]:
    x = signal(n, rate)
    g = math.gcd(rate, 16000)
    y = resample_poly(x.astype(np.float64), 16000 // g, rate // g)
    out["cases"].append({"rate": rate, "x": [float(v) for v in x], "y": [float(v) for v in y]})
json.dump(out, open("testdata/resample_poly_golden.json", "w"))
print({c["rate"]: len(c["y"]) for c in out["cases"]}, "scipy", scipy.__version__)

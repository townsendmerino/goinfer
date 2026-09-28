#!/usr/bin/env python3
"""Global least-squares fit of a 3-term model over GOINFER_DECODE_TIMING's real-token
gate+up/down split, across all three CPU int4 model sizes:

    time_ms = L * tau_ms                     (fixed cost per fork/join barrier, L barriers/token)
            + L * c_ms_per_elem * n           (serial scalar activation quantizer, n = its own K)
            + bytes_MB * (1/B_MB_per_ms)      (true DRAM streaming bandwidth)

gate+up: n = hidden_size (one shared quantization of h, amortized over the fused gate+up call)
down:    n = intermediate_size (down's own quantization of the SwiGLU-activated vector)

3 unknowns (tau, c, 1/B), 6 equations (2 phases x 3 model sizes) -- overdetermined, not a fit
to each model individually. Numbers are decode-timing-*.log's own (2026-09-28, aikit v1.50.1).
"""
import numpy as np

bpp = 0.5625  # bytes/param, int4 with binary16 group scales (aikit v1.50.0+)

models = [
    dict(name="0.5B", H=896,  I=4864,  L=24, gu_ms=6.92,  down_ms=4.70),
    dict(name="1.5B", H=1536, I=8960,  L=28, gu_ms=18.72, down_ms=11.19),
    dict(name="7B",   H=3584, I=18944, L=28, gu_ms=84.65, down_ms=46.35),
]

rows, targets, labels = [], [], []
for m in models:
    gu_bytes_MB = m['L']*2*m['H']*m['I']*bpp/1e6
    down_bytes_MB = m['L']*m['H']*m['I']*bpp/1e6
    rows.append([m['L'], m['L']*m['H'], gu_bytes_MB]); targets.append(m['gu_ms']); labels.append((m['name'], "gate+up"))
    rows.append([m['L'], m['L']*m['I'], down_bytes_MB]); targets.append(m['down_ms']); labels.append((m['name'], "down"))

A = np.array(rows, dtype=float)
b = np.array(targets, dtype=float)
x, *_ = np.linalg.lstsq(A, b, rcond=None)
tau_ms, c_ms_per_elem, invB = x
print(f"tau = {tau_ms*1000:.1f} us/barrier")
print(f"c   = {c_ms_per_elem*1e6:.3f} ns/element (serial scalar quantizer)")
print(f"B   = {1/invB:.2f} GB/s (true streaming bandwidth)")
print()
pred = A @ x
print(f"{'model':6} {'phase':8} {'measured_ms':>12} {'pred_ms':>10} {'resid%':>8}")
for (name, phase), meas, p in zip(labels, targets, pred):
    print(f"{name:6} {phase:8} {meas:12.2f} {p:10.2f} {(p-meas)/meas*100:8.2f}")
ss_res = np.sum((b-pred)**2)
ss_tot = np.sum((b-np.mean(b))**2)
print(f"\nR^2 = {1-ss_res/ss_tot:.4f}")

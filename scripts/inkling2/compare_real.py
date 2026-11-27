# Usage: python3 scripts/inkling2/compare_real.py <goinfer out dir> <llama.cpp dump dir>
# Differences goinfer's per-visit outputs (decoder/inkling2_real_test.go: visits.bin) against llama.cpp's per-slot l_out-<slot>.bin
# (harness on cb_eval) for the same prompt, visit by visit: llama.cpp keeps slot il for the visits of physical layer il outside
# the loop and for pass 0, and slots n_phys + (pass-1)*core + (layer-start) for the later passes.
import sys, struct, numpy as np
gi, ll = sys.argv[1], sys.argv[2]
NPHYS, CS, CE, NPASS = 24, 6, 18, 3
b = open(f"{gi}/visits.bin", "rb").read()
npos, nvis, hid = struct.unpack("<3q", b[:24])
v = np.frombuffer(b[24:], dtype=np.float32).reshape(npos, nvis, hid)
slots = list(range(CS))
for p in range(NPASS):
    slots += [ip if p == 0 else NPHYS + (p - 1) * (CE - CS) + (ip - CS) for ip in range(CS, CE)]
slots += list(range(CE, NPHYS))
assert len(slots) == nvis, (len(slots), nvis)
def rd(p):
    raw = open(p, "rb").read(); ne = np.frombuffer(raw[:32], dtype=np.int64)
    return np.frombuffer(raw[32:], dtype=np.float32).reshape(int(ne[1]), int(ne[0]))
worst = 1.0
for vi, s in enumerate(slots):
    r = rd(f"{ll}/l_out-{s}.bin")
    g = v[:, vi, :]
    cos = (g * r).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(r, axis=-1))
    worst = min(worst, cos.min())
    print(f"visit {vi:2d} (layer {s if s < NPHYS else CS + (s - NPHYS) % (CE - CS):2d}, slot {s:2d}): cosine min {cos.min():.6f} mean {cos.mean():.6f}  max|diff| {np.abs(g - r).max():.3e}  |ref| {np.abs(r).max():.2e}")
print(f"worst visit cosine {worst:.6f}")

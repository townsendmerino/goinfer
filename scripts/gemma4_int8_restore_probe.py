"""Probe 2 (offline): does the harness's int8-per-row RESTORAGE crush the per-group affine
reconstruction? That is the suspected reason fakeInt4WM's affine+f32act cell is only
semi-coherent while fieldfare (real affine-g64 Q4) is fully coherent.

For a real expert tensor [rows, cols]:
  recon_affine  = per-group affine 4-bit reconstruct (group g, along cols)   <- MLX/real Q4
  recon_restore = recon_affine then int8-per-ROW symmetric (what fakeInt4WM stores)  <- harness
If cos(recon_restore, recon_affine) ~ 1.0, the int8 restorage is harmless and the semi->full
gap is NOT this artifact. If it degrades (esp. rows with mixed-magnitude groups), confirmed.
"""
import json, numpy as np
from safetensors import safe_open

CKPT = "/home/francis/models/gemma-4-26b-a4b-it"
IDX = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
_o = {}
def load(n):
    sh = IDX[n]; _o.setdefault(sh, safe_open(f"{CKPT}/{sh}", framework="pt"))
    return _o[sh].get_tensor(n).float().numpy()

def affine_g(w, g):  # per-row, per-group [min,max]/15 affine 4-bit, along cols
    r, c = w.shape
    out = np.empty_like(w)
    for gs in range(0, c, g):
        ge = min(gs+g, c); blk = w[:, gs:ge]
        lo = blk.min(1, keepdims=True); hi = blk.max(1, keepdims=True)
        s = np.where(hi > lo, (hi-lo)/15.0, 1.0)
        code = np.clip(np.round((blk-lo)/s), 0, 15)
        out[:, gs:ge] = code*s + lo
    return out

def int8_row(w):  # per-row symmetric int8, exactly linalg.QuantizeInt8
    mx = np.abs(w).max(1, keepdims=True)
    s = np.where(mx > 0, mx/127.0, 1.0)
    return np.clip(np.round(w/s), -127, 127)*s

def rowcos(a, b):
    num = (a*b).sum(1); da = np.sqrt((a*a).sum(1)); db = np.sqrt((b*b).sum(1))
    return num/np.maximum(da*db, 1e-20)

for name, expert in [("layers.0.experts.down_proj", 0),
                     ("layers.0.experts.gate_up_proj", 0),
                     ("layers.15.experts.down_proj", 0),
                     ("layers.29.experts.down_proj", 0)]:
    t = load(f"model.language_model.{name}")           # [E, rows, cols]
    w = t[expert].astype(np.float64)                    # one expert [rows, cols]
    for g in (64, 32):
        ra = affine_g(w, g)
        rr = int8_row(ra)
        cos_ra_orig = rowcos(ra, w)                     # affine vs original
        cos_rr_ra   = rowcos(rr, ra)                    # restorage vs affine  <- the artifact
        cos_rr_orig = rowcos(rr, w)                     # restorage vs original (what the harness feeds)
        print(f"{name} e{expert} g{g:>2}  shape={w.shape}  "
              f"cos(affine,orig)={cos_ra_orig.mean():.5f}  "
              f"cos(restore,affine)={cos_rr_ra.mean():.6f} (min {cos_rr_ra.min():.5f})  "
              f"cos(restore,orig)={cos_rr_orig.mean():.5f}")
    print()

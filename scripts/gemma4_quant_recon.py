"""Offline int4 reconstruction experiment for gemma-4-26b-a4b-it (task Steps 1b + 3).
No decode, no runtime code — pure numpy on real weight tensors, matching aikit's exact
symmetric int4 (scale=maxAbs/7, q=round(w/s) in [-7,7], 15 levels) vs affine (min/max
zero-point, 16 levels) at group 32 and 64, with int8 as the high-precision reference."""
import numpy as np, torch, sys
from safetensors import safe_open

CKPT="/home/francis/models/gemma-4-26b-a4b-it"
IDX=__import__("json").load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
_open={}
def load(name):
    sh=IDX[name]
    if sh not in _open: _open[sh]=safe_open(f"{CKPT}/{sh}",framework="pt")
    return _open[sh].get_tensor(name).float().numpy()

def groups(w,g):  # [rows,cols] -> pad cols to multiple of g, reshape [rows, nG, g]
    r,c=w.shape; nG=(c+g-1)//g; pad=nG*g-c
    wp=np.pad(w,((0,0),(0,pad)),constant_values=0.0)
    return wp.reshape(r,nG,g), c, pad

def sym4(w,g):
    wg,c,pad=groups(w,g)
    mx=np.abs(wg).max(2,keepdims=True); s=np.where(mx>0,mx/7.0,1.0)
    q=np.clip(np.round(wg/s),-7,7); deq=(q*s).reshape(w.shape[0],-1)[:,:c]
    return deq
def affine4(w,g):
    wg,c,pad=groups(w,g)
    lo=wg.min(2,keepdims=True); hi=wg.max(2,keepdims=True)
    s=np.where(hi>lo,(hi-lo)/15.0,1.0)
    code=np.clip(np.round((wg-lo)/s),0,15); deq=(code*s+lo).reshape(w.shape[0],-1)[:,:c]
    return deq
def int8sym(w):  # per-row symmetric int8 (reference)
    mx=np.abs(w).max(1,keepdims=True); s=np.where(mx>0,mx/127.0,1.0)
    return np.clip(np.round(w/s),-127,127)*s

def err(w,deq):
    a,b=w.ravel().astype(np.float64),deq.ravel().astype(np.float64)
    cos=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    return cos, float(np.abs(a-b).max())

def skew(w,g):  # per-group asymmetry: mean/maxabs (0=symmetric)
    wg,_,_=groups(w,g); mx=np.abs(wg).max(2)+1e-12
    return float(np.abs(wg.mean(2)/mx).mean())

TENSORS=[
 ("tied embed/head","model.language_model.embed_tokens.weight",30000),
 ("expert gate_up (e0 L0)","model.language_model.layers.0.experts.gate_up_proj",None),
 ("expert down (e0 L0)","model.language_model.layers.0.experts.down_proj",None),
 ("dense mlp gate (L0)","model.language_model.layers.0.mlp.gate_proj.weight",None),
 ("attn q_proj (L0)","model.language_model.layers.0.self_attn.q_proj.weight",None),
]
print(f"{'tensor':26s} {'shape':16s} {'skew32':7s} {'sym4g32':9s} {'sym4g64':9s} {'aff4g32':9s} {'aff4g64':9s} {'int8':9s}")
for label,name,samp in TENSORS:
    w=load(name)
    if w.ndim==3: w=w[0]  # expert 0
    if samp and w.shape[0]>samp:
        idx=np.random.RandomState(0).choice(w.shape[0],samp,replace=False); w=w[idx]
    sh=f"{w.shape[0]}x{w.shape[1]}"
    cfgs={"sym4g32":sym4(w,32),"sym4g64":sym4(w,64),"aff4g32":affine4(w,32),"aff4g64":affine4(w,64),"int8":int8sym(w)}
    row=f"{label:26s} {sh:16s} {skew(w,32):7.3f} "
    for k in ["sym4g32","sym4g64","aff4g32","aff4g64","int8"]:
        cos,mx=err(w,cfgs[k]); row+=f"{cos:.5f}   "
    print(row)
    sys.stdout.flush()

# ---- Bias probe (Step: the mechanism cosine misses) ----
# Per-group MEAN reconstruction error (a DC bias that compounds directionally across the
# residual stream) and relative RMS error, for symmetric vs affine. Cosine is
# scale/direction-insensitive and hides a systematic bias; this exposes it.
def biasrms(w, deq, g):
    e = (deq.astype(np.float64) - w.astype(np.float64))
    eg,_,_ = groups(e, g); wg,_,_ = groups(w, g)
    scale = (np.abs(wg).max(2)+1e-12)
    mean_bias = float(np.abs(eg.mean(2)/scale).mean())   # |per-group mean error| / group scale
    rms = float(np.sqrt((e**2).mean()) / (np.sqrt((w.astype(np.float64)**2).mean())+1e-12))
    return mean_bias, rms
print("\n=== BIAS (|per-group mean err|/scale) and rel-RMS: symmetric vs affine ===")
print(f"{'tensor':26s} {'sym-g32 bias':13s} {'aff-g32 bias':13s} {'sym-g32 rms':12s} {'aff-g32 rms':12s}")
for label,name,samp in TENSORS[:3]:
    w=load(name)
    if w.ndim==3: w=w[0]
    if samp and w.shape[0]>samp:
        idx=np.random.RandomState(0).choice(w.shape[0],samp,replace=False); w=w[idx]
    sb,sr=biasrms(w,sym4(w,32),32); ab,ar=biasrms(w,affine4(w,32),32)
    print(f"{label:26s} {sb:13.5f} {ab:13.5f} {sr:12.5f} {ar:12.5f}")

"""Step B: is the affine gain from the zero-point, or just a better SCALE? Add
clipped-symmetric + MSE-optimal-symmetric to the table, and measure per-group range
asymmetry (|min| vs |max|) — if the range is asymmetric, affine wins by using [min,max]
rather than [-maxabs,maxabs] (wasting levels on an empty tail), which a symmetric scale
can't recover; if clipped-sym matches affine, no format change is needed."""
import numpy as np, torch
from safetensors import safe_open
CKPT="/home/francis/models/gemma-4-26b-a4b-it"
IDX=__import__("json").load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
_o={}
def load(n):
    sh=IDX[n]; _o.setdefault(sh,safe_open(f"{CKPT}/{sh}",framework="pt")); return _o[sh].get_tensor(n).float().numpy()
def grp(w,g):
    r,c=w.shape;nG=(c+g-1)//g;wp=np.pad(w,((0,0),(0,nG*g-c)));return wp.reshape(r,nG,g),c
def rms(w,deq):
    e=(deq-w).astype(np.float64);return float(np.sqrt((e**2).mean())/(np.sqrt((w.astype(np.float64)**2).mean())+1e-12))
def sym(w,g,clip=1.0):  # symmetric, scale = clip_percentile(|w|)/7
    wg,c=grp(w,g)
    mx=np.quantile(np.abs(wg),clip,axis=2,keepdims=True) if clip<1.0 else np.abs(wg).max(2,keepdims=True)
    s=np.where(mx>0,mx/7.0,1.0);return (np.clip(np.round(wg/s),-7,7)*s).reshape(w.shape[0],-1)[:,:c]
def sym_mse(w,g):  # grid-search the symmetric scale minimizing MSE per group
    wg,c=grp(w,g); mx=np.abs(wg).max(2,keepdims=True)
    best=None;bestE=None
    for f in np.linspace(0.55,1.0,19):  # fraction of maxabs
        s=np.where(mx>0,mx*f/7.0,1.0); deq=np.clip(np.round(wg/s),-7,7)*s
        e=((deq-wg)**2).sum(2,keepdims=True)
        if bestE is None: best,bestE=deq,e
        else: m=e<bestE; best=np.where(m,deq,best); bestE=np.where(m,e,bestE)
    return best.reshape(w.shape[0],-1)[:,:c]
def affine(w,g):
    wg,c=grp(w,g);lo=wg.min(2,keepdims=True);hi=wg.max(2,keepdims=True)
    s=np.where(hi>lo,(hi-lo)/15.0,1.0);return (np.clip(np.round((wg-lo)/s),0,15)*s+lo).reshape(w.shape[0],-1)[:,:c]
def rangeasym(w,g):
    wg,_=grp(w,g);return float((np.abs(np.abs(wg.min(2))-np.abs(wg.max(2)))/(np.abs(wg).max(2)+1e-12)).mean())
for label,name in [("expert gate_up","model.language_model.layers.0.experts.gate_up_proj"),("dense mlp","model.language_model.layers.0.mlp.gate_proj.weight")]:
    w=load(name)
    if w.ndim==3:w=w[0]
    print(f"\n{label} {w.shape}  range-asym(g32)={rangeasym(w,32):.3f}")
    print(f"  sym-g32 (maxabs)   rms={rms(w,sym(w,32)):.4f}")
    print(f"  sym-g32 clip99.9   rms={rms(w,sym(w,32,0.999)):.4f}")
    print(f"  sym-g32 clip99.5   rms={rms(w,sym(w,32,0.995)):.4f}")
    print(f"  sym-g32 MSE-scale  rms={rms(w,sym_mse(w,32)):.4f}")
    print(f"  affine-g32         rms={rms(w,affine(w,32)):.4f}")
    print(f"  sym-g64 (maxabs)   rms={rms(w,sym(w,64)):.4f}")

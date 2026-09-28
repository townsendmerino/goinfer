#!/usr/bin/env python3
"""Weight bytes streamed per decode token: a Q4_K_M GGUF as llama.cpp reads it (every 2-D tensor at its own
ggml type; token_embd only looked up when output.weight exists) vs goinfer int4 (int4 nibble + one f32 scale per
32-group = 0.625 B/param; the LM head int8 + one f32 per row; 1-D tensors f32). Header-only: reads no tensor data.
  python3 bytes_per_token.py A.gguf [B.gguf ...]"""
import hashlib, os, struct, sys
def rd(f, n): 
    b = f.read(n)
    assert len(b) == n, "short read"
    return b
def u32(f): return struct.unpack("<I", rd(f,4))[0]
def u64(f): return struct.unpack("<Q", rd(f,8))[0]
def s(f):
    n = u64(f); return rd(f,n).decode("utf-8","replace")
def val(f, t):
    if t==0: return rd(f,1)[0]
    if t==1: return struct.unpack("<b",rd(f,1))[0]
    if t==2: return struct.unpack("<H",rd(f,2))[0]
    if t==3: return struct.unpack("<h",rd(f,2))[0]
    if t==4: return u32(f)
    if t==5: return struct.unpack("<i",rd(f,4))[0]
    if t==6: return struct.unpack("<f",rd(f,4))[0]
    if t==7: return rd(f,1)[0]!=0
    if t==8: return s(f)
    if t==9:
        et=u32(f); n=u64(f); return [val(f,et) for _ in range(n)]
    if t==10: return u64(f)
    if t==11: return struct.unpack("<q",rd(f,8))[0]
    if t==12: return struct.unpack("<d",rd(f,8))[0]
    raise ValueError(f"bad kv type {t}")
def parse(path):
    f=open(path,"rb")
    assert rd(f,4)==b"GGUF", "not a GGUF"
    ver=u32(f); ntens=u64(f); nkv=u64(f)
    kv={}
    for _ in range(nkv):
        k=s(f); t=u32(f); kv[k]=val(f,t)
    tensors=[]
    for _ in range(ntens):
        name=s(f); nd=u32(f)
        dims=[u64(f) for _ in range(nd)]
        ttype=u32(f); off=u64(f)
        tensors.append((name,tuple(dims),ttype,off))
    align=kv.get("general.alignment",32)
    pos=f.tell()
    data_off=(pos+align-1)//align*align
    return {"ver":ver,"kv":kv,"tensors":tensors,"data_off":data_off,"path":path,"f":f}
def payload_md5(g, total_size):
    f=g["f"]; f.seek(g["data_off"])
    h=hashlib.md5(); left=total_size-g["data_off"]
    while left>0:
        b=f.read(min(1<<24,left))
        if not b: break
        h.update(b); left-=len(b)
    return h.hexdigest()
def sizes(g, total):
    ts=sorted(g["tensors"], key=lambda t:t[3])
    out={}
    for i,(name,dims,tt,off) in enumerate(ts):
        end = ts[i+1][3] if i+1<len(ts) else (total-g["data_off"])
        out[name]=(off,end-off,dims,tt)
    return out
def per_tensor_md5(path):
    g=parse(path); total=os.path.getsize(path)
    sz=sizes(g,total); f=g["f"]; res={}
    for name,(off,n,dims,tt) in sz.items():
        f.seek(g["data_off"]+off)
        h=hashlib.md5(); left=n
        while left>0:
            b=f.read(min(1<<24,left))
            if not b: break
            h.update(b); left-=len(b)
        res[name]=(h.hexdigest(),n,dims,tt)
    return res
# ggml type -> (block elems, block bytes)
T = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),30:(1,2)}
N = {0:"F32",1:"F16",2:"Q4_0",6:"Q5_0",8:"Q8_0",12:"Q4_K",13:"Q5_K",14:"Q6_K",30:"BF16"}
def tensors(path):
    f = open(path, "rb"); assert rd(f,4) == b"GGUF"; u32(f); nt = u64(f); nkv = u64(f)
    for _ in range(nkv): s(f); t = u32(f); val(f, t)
    out = []
    for _ in range(nt):
        name = s(f); nd = u32(f); dims = [u64(f) for _ in range(nd)]; t = u32(f); u64(f)
        n = 1
        for d in dims: n *= d
        be, bb = T[t]; out.append((name, dims, t, n, n // be * bb))
    return out
for p in sys.argv[1:]:
    ts = tensors(p)
    tied = not any(n == "output.weight" for n, *_ in ts)
    g_bytes = 0; i_bytes = 0; mix = {}
    for name, dims, t, n, b in ts:
        if name == "token_embd.weight" and not tied: continue      # one row per token: negligible
        if len(dims) < 2: g_bytes += b; i_bytes += n * 4; continue   # norms/biases: f32 in goinfer
        head = name in ("output.weight",) or (tied and name == "token_embd.weight")
        g_bytes += b
        i_bytes += (n + dims[1] * 4) if head else (n // 2 + n // 32 * 4)  # head int8 + f32/row; else int4 nibble + f32 per 32
        mix[N.get(t, t)] = mix.get(N.get(t, t), 0) + b
    print(f"{p.split('/')[-1]}: tied={tied}  per-token bytes  Q4_K_M {g_bytes/1e6:.0f} MB  goinfer (int4, int8 head) {i_bytes/1e6:.0f} MB  ratio {i_bytes/g_bytes:.3f}")
    print("   Q4_K_M by type (MB):", {k: round(v/1e6) for k, v in mix.items()})

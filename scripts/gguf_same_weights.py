#!/usr/bin/env python3
"""gguf_same_weights.py — prove two GGUF files carry the SAME WEIGHTS.

WHY THIS EXISTS. bench_peer.py's docstring promised "the same GGUF file on both sides
(verify by md5 before trusting a run)". That check is UNSATISFIABLE through Ollama's import
path, for any model: `ollama create` repacks the container in a different TENSOR ORDER, so
every tensor offset changes and the file md5 (and any tail slice of it) always differs even
when the weights are bit-identical. Measured on qwen2.5-7b-instruct-q4_k_m, 2026-08-22:
metadata keys and values identical, 339/339 tensor names/shapes/types identical, all 339
OFFSETS different, whole-file md5 different, per-tensor payload 339/339 IDENTICAL.

So the achievable guarantee is per-tensor: read each tensor at ITS OWN offset in ITS OWN
file and compare. That is what this does, and it is what the peer benchmark's fairness
actually rests on.

  python3 scripts/gguf_same_weights.py A.gguf B.gguf
"""
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

a=sys.argv[1]
b=sys.argv[2]
A=per_tensor_md5(a); B=per_tensor_md5(b)
print(f"tensors A={len(A)} B={len(B)}")
names=set(A)&set(B)
same=sum(1 for n in names if A[n][0]==B[n][0])
diff=[n for n in names if A[n][0]!=B[n][0]]
sizemismatch=[n for n in names if A[n][1]!=B[n][1]]
print(f"identical tensors : {same}/{len(names)}")
print(f"differing tensors : {len(diff)}")
print(f"size mismatches   : {len(sizemismatch)}")
for n in diff[:6]:
    print(f"  DIFF {n}: A(md5={A[n][0][:12]} bytes={A[n][1]}) B(md5={B[n][0][:12]} bytes={B[n][1]}) dims={A[n][2]} type={A[n][3]}")
print("\nVERDICT:", "SAME WEIGHTS (repacked container)" if not diff else "WEIGHTS DIFFER")

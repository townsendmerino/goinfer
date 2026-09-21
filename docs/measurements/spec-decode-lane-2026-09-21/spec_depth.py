import json,os,subprocess,sys,time,urllib.request,signal
BIN=os.path.expanduser("~/bench-r7b/serve-cuda-fa2")
MODEL=os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
SRC=open(os.path.expanduser("~/mycode/goinfer/decoder/sampler_selection_test.go")).read()
def prompt(ntok_chars):
    return "Here is a Go file:\n\n"+SRC[:ntok_chars]+"\n\nRewrite the file above EXACTLY, character for character, with no changes and no commentary."
def run(label,env,extra,chars,ntok=192):
    e=dict(os.environ); e.pop("GOINFER_CUDA_FLASH_DECODE",None); e.pop("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS",None); e.update(env)
    err=open(os.path.expanduser(f"~/bench-r7b/spec/{label}.err"),"w")
    p=subprocess.Popen([BIN,"-model","bench="+MODEL,"-backend","cuda","-quant","int4","-addr","127.0.0.1:18111","-ctx","8192"]+extra,env=e,stdout=subprocess.DEVNULL,stderr=err,preexec_fn=os.setsid)
    try:
        for _ in range(180):
            try: urllib.request.urlopen("http://127.0.0.1:18111/v1/models",timeout=1); break
            except Exception:
                if p.poll() is not None: return None,"server exited: "+open(os.path.expanduser(f"~/bench-r7b/spec/{label}.err")).read()[-300:]
                time.sleep(1)
        def one():
            body={"model":"bench","stream":True,"max_tokens":ntok,"temperature":0,"stream_options":{"include_usage":True},"messages":[{"role":"user","content":prompt(chars)}]}
            req=urllib.request.Request("http://127.0.0.1:18111/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"})
            ts=[];txt=""
            for l in urllib.request.urlopen(req,timeout=600):
                l=l.decode().strip()
                if l.startswith("data:") and "[DONE]" not in l:
                    d=json.loads(l[5:])
                    if d.get("choices") and d["choices"][0]["delta"].get("content"): ts.append(time.time()); txt+=d["choices"][0]["delta"]["content"]
            return (len(ts)-1)/(ts[-1]-ts[0]),txt
        one()
        rs=[];tx=None
        for _ in range(3):
            r,t=one(); rs.append(r); tx=t
        return rs,tx
    finally:
        try: os.killpg(p.pid,signal.SIGTERM)
        except Exception: pass
        time.sleep(4)
if __name__=="__main__":
    chars=int(sys.argv[1])
    out={}
    for label,env,extra in [("exact",{},[]),("lane",{"GOINFER_CUDA_FLASH_DECODE":"16"},[]),("exact+spec",{},["-spec","ngram"]),("lane+spec",{"GOINFER_CUDA_FLASH_DECODE":"16"},["-spec","ngram"])]:
        rs,tx=run(label,env,extra,chars)
        if rs is None: print(label,"FAILED TO START:",tx.strip()[-200:]); continue
        out[label]=tx
        print(f"{label:11s} chars={chars} tok/s {[round(x,1) for x in rs]}  text[:60]={tx[:60]!r}",flush=True)
    if "exact" in out and "exact+spec" in out: print("exact vs exact+spec output identical:",out["exact"]==out["exact+spec"])
    if "exact" in out and "lane" in out: print("exact vs lane output identical:",out["exact"]==out["lane"])

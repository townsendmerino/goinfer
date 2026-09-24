"""bench_sidecar_cuda.py — CUDA serve: .giw sidecar default vs -direct-load (docs/measurements/cpu-giw-vs-direct-2026-09-24.md,
"CUDA"). Fresh server per arm, ABBA x2 per model, 3 greedy 256-token decodes per server; decode tok/s = 256 / (t(257) - t(1)).
GOINFER_SERVE_CUDA or the scratch path below names the binary."""
import json,os,signal,socket,subprocess,sys,time,urllib.request
SP=os.environ.get("BENCH_OUT", "/tmp")
SERVE=os.environ.get("GOINFER_SERVE_CUDA", SP+"/serve-sidecar"); PORT=8095; M=os.path.expanduser("~/models/")
MODELS={"1.5B":M+"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf","7B":M+"qwen2.5-7b-instruct-q4_k_m.gguf"}
PROMPT="Write a detailed explanation of how a token-bucket rate limiter works, with an example in Go."
def up():
    t=time.time()
    while time.time()-t<900:
        try:
            with socket.create_connection(("127.0.0.1",PORT),1): return time.time()-t
        except OSError: time.sleep(0.2)
    raise RuntimeError("no server")
def post(p):
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    t=time.perf_counter(); b=json.loads(urllib.request.urlopen(r,timeout=900).read()); return time.perf_counter()-t,b
def arm(model,direct,log):
    argv=[SERVE,"-model","m="+MODELS[model],"-backend","cuda","-addr",f"127.0.0.1:{PORT}","-quant","int4"]+(["-direct-load"] if direct else [])
    p=subprocess.Popen(argv,stdout=log,stderr=log,preexec_fn=os.setsid)
    try:
        load=up(); res=[]
        base={"model":"m","messages":[{"role":"user","content":PROMPT}],"temperature":0}
        post({**base,"max_tokens":8})  # warm
        for i in range(3):
            t1,_=post({**base,"max_tokens":1}); t2,b=post({**base,"max_tokens":257})
            n=b["usage"]["completion_tokens"]
            res.append({"decode_tok_s":(n-1)/(t2-t1),"text":b["choices"][0]["message"]["content"],"n":n})
        return load,res
    finally:
        os.killpg(os.getpgid(p.pid),signal.SIGTERM); p.wait(60); time.sleep(4)
out={}
for model in ("1.5B","7B"):
    log=open(f"{SP}/server-{model}.log","ab")
    # first sidecar start transcodes (cold); recorded separately, not a timing arm
    t=time.time(); l,_=arm(model,False,log); out[model+"_first_start_s"]=round(time.time()-t,1); out[model+"_first_load_s"]=round(l,1)
    rows=[]
    for order in (("side","direct"),("direct","side"),("side","direct"),("direct","side")):
        for a in order:
            l,r=arm(model,a=="direct",log)
            rows.append({"arm":a,"load_s":round(l,2),"tok_s":[round(x["decode_tok_s"],2) for x in r],"texts":[x["text"] for x in r],"n":[x["n"] for x in r]})
            print(model,a,rows[-1]["load_s"],rows[-1]["tok_s"],flush=True)
    out[model]=rows
json.dump(out,open(f"{SP}/result.json","w"),indent=1)
print("DONE",flush=True)

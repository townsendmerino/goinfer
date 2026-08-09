import json, os, signal, subprocess, socket, time, urllib.request, sys
SP="/tmp/claude-1000/-home-francis-mycode-goinfer/a5a0a129-36b9-49b1-a9e4-82d056df5e45/scratchpad"
MODELS={"0.5B":os.path.expanduser("~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
        "1.5B":os.path.expanduser("~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")}
PORT=8099
def wait(p,t=180):
    t0=time.time()
    while time.time()-t0<t:
        try:
            with socket.create_connection(("127.0.0.1",p),timeout=1): return True
        except OSError: time.sleep(0.5)
    return False
def ptoks(text):
    pl={"model":"bench","stream":False,"max_tokens":1,"temperature":0,
        "messages":[{"role":"user","content":text}]}
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(pl).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=120) as resp:
        return json.load(resp)["usage"]["prompt_tokens"]
# " the" is a single common token; repetition gives ~1 token/word.
def make(n): return "Continue this text. " + " ".join(["the"]*n)
out={}
for mk,path in MODELS.items():
    p=subprocess.Popen([f"{SP}/goinfer-serve","-model",f"bench={path}","-backend","cuda",
        "-addr",f"127.0.0.1:{PORT}","-quant","int4"],stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,preexec_fn=os.setsid)
    try:
        assert wait(PORT), "server down"
        for target in (128,512,2048,3900):
            n=target; best=None
            for _ in range(12):
                t=ptoks(make(n))
                if best is None or abs(t-target)<abs(best[1]-target): best=(n,t)
                if abs(t-target)<=max(3,target*0.01): break
                n=max(1,int(n*target/max(t,1)))
            out[f"{mk}:{target}"]={"words":best[0],"tokens":best[1],"text":make(best[0])}
            print(f"{mk} depth={target}: {best[1]} tokens ({best[0]} words)",flush=True)
    finally:
        os.killpg(os.getpgid(p.pid),signal.SIGTERM); p.wait(timeout=30); time.sleep(3)
json.dump(out,open(f"{SP}/prompts.json","w"))
print("saved",len(out),"prompts")

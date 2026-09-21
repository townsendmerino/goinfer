import json,os,subprocess,sys,time,urllib.request,signal
chars=int(sys.argv[1]); lane=sys.argv[2]=="lane"; skip=sys.argv[3]; count=sys.argv[4]; tag=sys.argv[5]
BIN=os.path.expanduser("~/bench-r7b/serve-cuda-fa3")
MODEL=os.path.expanduser("~/models/qwen2.5-7b-instruct-q4_k_m.gguf")
SRC=open(os.path.expanduser("~/mycode/goinfer/decoder/model.go")).read()
prompt="Here is a Go file:\n\n"+SRC[:chars]+"\n\nSummarize what the file above does in detail."
env=dict(os.environ); env.pop("GOINFER_CUDA_FLASH_DECODE",None)
if lane: env["GOINFER_CUDA_FLASH_DECODE"]="16"
out=open(os.path.expanduser(f"~/bench-r7b/d7/ncu_{tag}.csv"),"w")
cmd=["ncu","--metrics","gpu__time_duration.sum","--csv","--page","raw","--launch-skip",skip,"--launch-count",count,
     BIN,"-model","bench="+MODEL,"-backend","cuda","-quant","int4","-addr","127.0.0.1:18131","-ctx","8192"]
p=subprocess.Popen(cmd,stdout=out,stderr=subprocess.DEVNULL,preexec_fn=os.setsid,env=env)
try:
    for _ in range(400):
        try: urllib.request.urlopen("http://127.0.0.1:18131/v1/models",timeout=1); break
        except Exception: time.sleep(1)
    body={"model":"bench","stream":True,"max_tokens":48,"temperature":0,"messages":[{"role":"user","content":prompt}]}
    for _ in range(2):
        req=urllib.request.Request("http://127.0.0.1:18131/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"})
        for l in urllib.request.urlopen(req,timeout=3000): pass
finally:
    try: os.killpg(p.pid,signal.SIGTERM)
    except Exception: pass
    time.sleep(6)

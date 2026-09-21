import json,os,subprocess,sys,time,urllib.request,signal
sys.path.insert(0,os.path.expanduser("~/bench-r7b/spec"))
import spec_depth as sd
BIN=sd.BIN; MODEL=sd.MODEL
env=dict(os.environ)
cmd=["ncu","--metrics","gpu__time_duration.sum","--csv","--page","raw","--launch-skip",sys.argv[1],"--launch-count",sys.argv[2],
     BIN,"-model","bench="+MODEL,"-backend","cuda","-quant","int4","-addr","127.0.0.1:18112","-ctx","8192","-spec","ngram"]
out=open(os.path.expanduser("~/bench-r7b/spec/ncu_spec.csv"),"w")
p=subprocess.Popen(cmd,stdout=out,stderr=subprocess.DEVNULL,preexec_fn=os.setsid)
try:
    for _ in range(300):
        try: urllib.request.urlopen("http://127.0.0.1:18112/v1/models",timeout=1); break
        except Exception: time.sleep(1)
    body={"model":"bench","stream":True,"max_tokens":192,"temperature":0,"messages":[{"role":"user","content":sd.prompt(7000)}]}
    for _ in range(2):
        req=urllib.request.Request("http://127.0.0.1:18112/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"})
        for l in urllib.request.urlopen(req,timeout=1800): pass
finally:
    try: os.killpg(p.pid,signal.SIGTERM)
    except Exception: pass
    time.sleep(5)

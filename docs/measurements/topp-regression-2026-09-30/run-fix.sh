#!/bin/bash
# Top-p fix, served check on nobara-pc (docs/measurements/topp-regression-2026-09-30.md). One session, interleaved.
set -u
W=$HOME/goinfer-bench/topp-regression-2026-09-30
cd "$HOME/goinfer-bench/peer-sweep-2026-09-29/wt-754f12d3" || exit 1
idle() { for i in $(seq 1 60); do awk '{exit !($1<0.8)}' /proc/loadavg && return; sleep 10; done; }
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_DEPTHS=none BENCH_MODELS=0.5B
export BENCH_CONFIGS=greedy,temp0.8_topp0.95 GOINFER_SERVE_CUDA=$W/serve-cuda-fix
idle; echo "$(date '+%H:%M:%S') fix vs 411e7fc4"; BENCH_SERVE_LOG_DIR=$W/serve-logs-fix1 GOINFER_SERVE_CUDA_OLD=$HOME/bench-peer-claim/serve-cuda-411e7fc4 python3 -u scripts/bench_peer.py "$W/fix-vs-411e7fc4.json"; echo "exit=$?"
idle; echo "$(date '+%H:%M:%S') fix vs 754f12d3"; BENCH_SERVE_LOG_DIR=$W/serve-logs-fix2 GOINFER_SERVE_CUDA_OLD=$HOME/goinfer-bench/peer-sweep-2026-09-29/serve-cuda-754f12d3 python3 -u scripts/bench_peer.py "$W/fix-vs-754f12d3.json"; echo "exit=$?"
# identity: the same seeded top-p request to each binary; the reply text must match
idle; echo "$(date '+%H:%M:%S') identity"
python3 - "$W" <<'PY'
import json,os,signal,socket,subprocess,sys,time,urllib.request
W=sys.argv[1]; M=os.path.expanduser("~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
bins={"fix":W+"/serve-cuda-fix","411e7fc4":os.path.expanduser("~/bench-peer-claim/serve-cuda-411e7fc4"),"754f12d3":os.path.expanduser("~/goinfer-bench/peer-sweep-2026-09-29/serve-cuda-754f12d3")}
out={}
for name,b in bins.items():
    p=subprocess.Popen([b,"-model","bench="+M,"-backend","cuda","-addr","127.0.0.1:8196","-quant","int4"],stdout=subprocess.DEVNULL,stderr=open(W+f"/identity-{name}.log","w"),preexec_fn=os.setsid)
    t=time.time()
    while time.time()-t<180:
        try: socket.create_connection(("127.0.0.1",8196),1).close(); break
        except OSError: time.sleep(0.3)
    texts=[]
    for seed in (1,2,3,4,5):
        body={"model":"bench","messages":[{"role":"user","content":"Write a short Go function that reverses a string, and explain it."}],"temperature":0.8,"top_p":0.95,"seed":seed,"max_tokens":128}
        r=urllib.request.Request("http://127.0.0.1:8196/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        texts.append(json.loads(urllib.request.urlopen(r,timeout=300).read())["choices"][0]["message"]["content"])
    out[name]=texts
    os.killpg(os.getpgid(p.pid),signal.SIGTERM); p.wait(30); time.sleep(3)
json.dump(out,open(W+"/identity.json","w"),indent=1)
for a in ("411e7fc4","754f12d3"):
    same=sum(x==y for x,y in zip(out["fix"],out[a]))
    print(f"fix vs {a}: {same}/5 seeded top-p replies identical")
PY
echo "$(date '+%H:%M:%S') DONE"

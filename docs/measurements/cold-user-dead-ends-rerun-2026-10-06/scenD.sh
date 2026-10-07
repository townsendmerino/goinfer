#!/bin/bash
# usage: scenD.sh <label> <serve-binary> <model.gguf> — the README's -moe-cache-experts command on the 8 GB card, under the amended swap-safety rule
LABEL=$1; BIN=$2; M=$3; D=/tmp/claude-1000/scen; OUT=$D/D-$LABEL; mkdir -p $OUT
echo "[$(date +%T)] $LABEL: $($BIN --version 2>&1 | head -1); swap before: $(free -m | awk '/Swap/{print $3" MB used"}'); avail $(free -m | awk '/Mem/{print $7}') MB"
(setsid nohup $BIN -addr 127.0.0.1:18085 -backend cuda -moe-cache-experts -model $M > $OUT/serve.log 2>&1 < /dev/null & echo $! > $OUT/serve.pid)
PID=$(cat $OUT/serve.pid); python3 $D/swapwatch.py $PID $OUT/swap.csv 512 3000 &
WPID=$!
t0=$(date +%s)
for i in $(seq 1 400); do curl -s -o /dev/null http://127.0.0.1:18085/v1/models && break; [ -e /proc/$PID ] || { echo "server exited before ready"; break; }; sleep 3; done
echo "[$(date +%T)] ready after $(( $(date +%s)-t0 )) s (or exited)"
[ -e /proc/$PID ] && { sed 's/\x1b\[[0-9;]*m//g' $OUT/serve.log | grep -iE "decode path|context:|fit:|swap guard|moe|serving on|warning|not armed|C′|cache" | head -14 | cut -c1-200
  M2=$(curl -s http://127.0.0.1:18085/v1/models | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")
  echo "model id: $M2"
  python3 - "$M2" <<'P'
import json,sys,time,urllib.request
m=sys.argv[1]
for q in ["What is the capital of France? Answer in one sentence.","List three prime numbers greater than 100."]:
    body=json.dumps({"model":m,"stream":True,"temperature":0,"max_tokens":48,"messages":[{"role":"user","content":q}]}).encode()
    t0=time.time(); r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18085/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=900)
    first=None;n=0;txt=""
    for line in r:
        line=line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        d=json.loads(line[5:]); ch=d.get("choices") or []
        if ch and ch[0].get("delta",{}).get("content"):
            if first is None: first=time.time()-t0
            n+=1; txt+=ch[0]["delta"]["content"]
    tot=time.time()-t0
    print(f"chat: TTFT {first:.2f}s, {n} chunks in {tot:.1f}s ({(n-1)/max(tot-first,1e-9):.1f} chunks/s) -> {txt[:90]!r}")
P
  echo "[$(date +%T)] serve check"; timeout 900 $BIN check http://127.0.0.1:18085 > $OUT/check.txt 2>&1; sed 's/\x1b\[[0-9;]*m//g' $OUT/check.txt | grep -E "ok|FAIL|skip|passed" | cut -c1-130; }
kill $PID 2>/dev/null; sleep 4; kill $WPID 2>/dev/null; wait $WPID 2>/dev/null
tail -3 $OUT/swap.csv | cut -c1-150; echo "[$(date +%T)] $LABEL done"

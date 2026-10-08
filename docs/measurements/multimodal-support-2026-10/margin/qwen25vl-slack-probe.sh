#!/usr/bin/env bash
# Untimed correctness probe: serve defaults on Qwen2.5-VL 3B, one ~8000-patch image then a small one; does the tower stay on CUDA with no OOM?
set -uo pipefail
S=$1; ARM=$2; PORT=$3
M=$HOME/models/qwen25vl-3b-instruct
SMALL=$HOME/mycode/goinfer/testdata/gemma3_preprocess_image.png
$S/serve-$ARM --model $M --backend cuda --addr 127.0.0.1:$PORT > $S/$ARM.serve.log 2>&1 < /dev/null &
pid=$!
for _ in $(seq 1 180); do curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break; kill -0 $pid 2>/dev/null || { echo "serve exited"; exit 1; }; sleep 1; done
echo "[$ARM] ready; free VRAM after load: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used"
python3 - $PORT $S/big.png $SMALL $ARM <<'PY'
import base64, json, sys, time, urllib.request, subprocess
port, big, small, arm = sys.argv[1:]
def used(): return subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
for name, path in (("big", big), ("small", small), ("big-again", big)):
    b64 = base64.b64encode(open(path,"rb").read()).decode()
    body = {"model":"qwen25vl-3b-instruct","stream":False,"max_tokens":8,"temperature":0,"messages":[{"role":"user","content":[{"type":"text","text":"Describe this image."},{"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}}]}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})
    t0=time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r: d=json.loads(r.read())
        out = d["choices"][0]["message"]["content"][:50].replace("\n"," ")
        print(f"[{arm}] {name}: HTTP 200 in {time.time()-t0:.1f}s (untimed, exploratory), prompt_tokens={d['usage']['prompt_tokens']}, used VRAM {used()} MiB, reply: {out!r}")
    except Exception as e:
        print(f"[{arm}] {name}: FAILED {e}; used VRAM {used()} MiB")
PY
kill $pid 2>/dev/null; wait $pid 2>/dev/null
echo "--- [$ARM] serve log (plan / tower / fallback lines):"
grep -iE "resident|tower|encoder|fall|trim|slot|reserved|decode path|out of memory|OOM|declin|cpu" $S/$ARM.serve.log | cut -c1-230 | head -14

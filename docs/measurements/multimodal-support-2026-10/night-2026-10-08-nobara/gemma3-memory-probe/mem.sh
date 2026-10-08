#!/usr/bin/env bash
# Untimed memory probe: Gemma 3 4B, one arm per call. Reads VRAM idle / after load / after the first image / after a second image.
S=$1; ARM=$2; shift 2
PORT=18781
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
echo "[$ARM] idle: $(used) MiB"
$S/serve-main --model $HOME/models/gemma-3-4b-it --backend cuda --addr 127.0.0.1:$PORT "$@" > $S/$ARM.serve.log 2>&1 < /dev/null &
pid=$!
for _ in $(seq 1 180); do curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break; kill -0 $pid 2>/dev/null || { echo serve exited; exit 1; }; sleep 1; done
echo "[$ARM] after load: $(used) MiB"
python3 - $PORT $HOME/mycode/goinfer/testdata/gemma3_preprocess_image.png $ARM <<'PY'
import base64, json, sys, urllib.request, subprocess
port, img, arm = sys.argv[1:]
def used(): return subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
b64 = base64.b64encode(open(img,"rb").read()).decode()
for i in (1,2):
    # a fresh image each time: flip one pixel row via a nonce in the prompt is not enough for the tower cache, so alter the PNG bytes' trailing metadata
    raw = base64.b64decode(b64) + bytes([i])*0
    body = {"model":"gemma-3-4b-it","stream":False,"max_tokens":6,"temperature":0,"messages":[{"role":"user","content":[{"type":"text","text":f"Describe this image. ({i})"},{"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}}]}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r: d=json.loads(r.read())
        print(f"[{arm}] image request {i}: HTTP 200, prompt_tokens={d['usage']['prompt_tokens']}, used {used()} MiB, reply {d['choices'][0]['message']['content'][:40]!r}")
    except Exception as e:
        print(f"[{arm}] image request {i}: FAILED {e}; used {used()} MiB")
PY
kill $pid 2>/dev/null; wait $pid 2>/dev/null
grep -iE "resident context|KV slots|encoder|less .* reserved|tower" $S/$ARM.serve.log | head -4 | cut -c1-230

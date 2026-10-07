#!/bin/bash
# G4c (docs/tasks/task-multimodal-support-2026-10.md, "S1 on CUDA"): F2b's request through ONE serve binary, resident arm --backend cuda, CPU arm --backend cpu.
# usage: g4c.sh <arm: cuda|cpu> <port>
ARM=$1; PORT=$2; G=/tmp/claude-1000/g4c; R=$HOME/mycode/goinfer
M=$HOME/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf; V=$HOME/models/gemma-4-E2B-unq
setsid nohup $G/serve-cuda -addr 127.0.0.1:$PORT -backend $ARM -model $M -vision $V > $G/serve-$ARM.log 2>&1 < /dev/null & echo $! > $G/$ARM.pid
for i in $(seq 1 300); do curl -s -o /dev/null http://127.0.0.1:$PORT/v1/models && break; sleep 2; done
python3 - $PORT $R/testdata/glm_ocr/table.png > $G/reply-$ARM.json <<'P'
import base64,json,sys,urllib.request
port,img=sys.argv[1],sys.argv[2]
b=base64.b64encode(open(img,'rb').read()).decode()
mid=json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"))['data'][0]['id']
body={"model":mid,"temperature":0,"max_tokens":32,"messages":[{"role":"user","content":[{"type":"text","text":"What does this image show? Answer briefly."},{"type":"image_url","image_url":{"url":"data:image/png;base64,"+b}}]}]}
r=urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900)
print(r.read().decode())
P
python3 -c "import json;d=json.load(open('$G/reply-$ARM.json'));print(d['choices'][0]['message']['content'])" > $G/reply-$ARM.txt
kill $(cat $G/$ARM.pid); sleep 3

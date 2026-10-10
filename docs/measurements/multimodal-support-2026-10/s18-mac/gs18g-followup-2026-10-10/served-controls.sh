#!/usr/bin/env bash
# Exploratory controls for G-S18g, by day: same binary, images, request; arms f16 (reference), cpuf32 (the true f32 tower on the CPU), int8.
set -uo pipefail
SERVE=$HOME/goinfer-bench/s18/serve-metal-c2891555
DIR=$HOME/models/gemma-3-4b-it
OUT=$1
cd $HOME/goinfer-bench/s18/wt-c2891555 || exit 2
PORT=18461
for arm in f16 cpuf32 int8; do
  case $arm in f16) fl=(-vision-quant f32);; cpuf32) fl=(-vision-device cpu -vision-quant f32);; int8) fl=(-vision-quant int8);; esac
  echo "[$(date +%T)] arm $arm: ${fl[*]}"
  "$SERVE" --model "$DIR" --backend metal "${fl[@]}" --addr 127.0.0.1:$PORT > "$OUT/serve-$arm.log" 2>&1 </dev/null &
  pid=$!
  for _ in $(seq 1 600); do
    curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break
    kill -0 $pid 2>/dev/null || { echo "serve exited ($arm)"; break; }
    sleep 1
  done
  python3 - "$OUT" "$arm" "$PORT" <<'PY'
import base64, json, sys, time, urllib.request
out, arm, port = sys.argv[1:4]
imgs = ["glm_ocr/table.png", "gemma3_preprocess_image.png", "qwen25vl_preprocess_image.png", "glm_ocr/formula.png"]
recs = {}
for im in imgs:
    b = base64.b64encode(open("testdata/" + im, "rb").read()).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}},
        {"type": "text", "text": "What does this image show? Answer briefly."}]}],
        "max_tokens": 32, "temperature": 0, "logprobs": True, "top_logprobs": 3}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    c = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]
    recs[im] = {"reply": c["message"]["content"], "logprobs": (c.get("logprobs") or {}).get("content") or []}
    print(f"  {arm} {im} ({time.time() - t:.1f}s): {c['message']['content'][:110]!r}", flush=True)
json.dump(recs, open(f"{out}/replies-{arm}.json", "w"))
PY
  grep -E "decode path|loaded vision tower|encoder|int8 Metal|note:" "$OUT/serve-$arm.log" | cut -c1-220
  kill $pid; wait $pid 2>/dev/null || true; sleep 8
done
echo "[$(date +%T)] done"

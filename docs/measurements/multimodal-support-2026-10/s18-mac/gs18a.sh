#!/usr/bin/env bash
# G-S18a, the Mac cell (by day, a served correctness check, not timed). Run from the repo root.
set -uo pipefail
SERVE=$1 OUT=$2
mkdir -p "$OUT"
DIR=$HOME/models/gemma-3-4b-it
GIW=$HOME/models/gemma-3-4b-it.int4.metal.giw
PORT=18461
budget() { vm_stat | awk -v ps="$(pagesize)" '/Pages (free|inactive|purgeable|speculative)/ {gsub("\\.","",$NF); n+=$NF} END {printf "%.2f GB reclaimable", n*ps/1e9}'; }
arm() { # name, serve args...
  local name=$1; shift
  echo "[$(date +%T)] arm $name: $* ; $(budget); swap $(sysctl -n vm.swapusage | awk '{print $6}')" | tee -a "$OUT/provenance.txt"
  "$SERVE" "$@" --addr 127.0.0.1:$PORT > "$OUT/serve-$name.log" 2>&1 </dev/null &
  local pid=$!
  for _ in $(seq 1 600); do
    curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break
    kill -0 $pid 2>/dev/null || { echo "serve exited ($name)"; return 1; }
    sleep 1
  done
  python3 - "$OUT" "$name" "$PORT" <<'PY'
import base64, json, sys, time, urllib.request
out, name, port = sys.argv[1:4]
img = base64.b64encode(open("testdata/glm_ocr/table.png", "rb").read()).decode()
body = {"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
    {"type": "text", "text": "What does this image show? Answer briefly."}]}],
    "max_tokens": 32, "temperature": 0, "logprobs": True, "top_logprobs": 3}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t = time.time()
c = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]
json.dump({"reply": c["message"]["content"], "logprobs": (c.get("logprobs") or {}).get("content") or []}, open(f"{out}/reply-{name}.json", "w"))
print(f"  {name} ({time.time() - t:.1f}s): {c['message']['content'][:140]!r}")
PY
  grep -E "decode path|loaded vision tower|int8 Metal|KV plan|sidecar|resident build|declin" "$OUT/serve-$name.log" | cut -c1-220
  kill $pid; wait $pid 2>/dev/null || true; sleep 30
}
arm default --model "$DIR" --backend metal || exit 1
vq=f32; grep -q "encoder int8" "$OUT/serve-default.log" && vq=int8
echo "default arm's tower: $vq" | tee -a "$OUT/provenance.txt"
arm handset --model "$GIW" --vision "$DIR" --backend metal -ctx 2048 -kv-sessions 1 -vision-quant "$vq" || exit 1
python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
d, h = (json.load(open(f"{out}/reply-{a}.json")) for a in ("default", "handset"))
same = d["reply"] == h["reply"] and [x["token"] for x in d["logprobs"]] == [x["token"] for x in h["logprobs"]]
print("G-S18a replies:", "IDENTICAL" if same else "DIFFER")
PY

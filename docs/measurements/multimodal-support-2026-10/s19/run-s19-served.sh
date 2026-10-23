#!/usr/bin/env bash
# S19's served runs (docs/tasks/task-multimodal-support-2026-10.md, "S19"): the real Gemma 4 26B-A4B with an image, served as
# Francis serves it (-backend cuda -moe-cache-experts --ctx 16384, the 560-soft-token vision dir), one fresh server per arm,
# the same requests on every arm: each image in IMAGES, "Transcribe every field of this form as key: value lines.",
# greedy, MAXTOK tokens, top-3 log-probabilities. Each request's wall time, reply and logprobs are kept, with the server's
# "vision: decoded ..." line naming where the turn prefilled.
#
# Usage (from the repo root): run-s19-served.sh <out dir> <label>=<serve binary>[:<env>] ...
#   e.g. run-s19-served.sh ~/goinfer-logs/g4-bidir/g3 new=$NEW ctl=$NEW:GOINFER_BATCHED_PREFILL=0
# MODEL, VISION, IMAGES (space-separated), MAXTOK, PORT, SETTLE override the defaults below.
set -uo pipefail
OUT=${1:?}; shift
MODEL=${MODEL:-$HOME/models/gemma4-26b-int4.giw}
VISION=${VISION:-$HOME/models/gemma-4-26b-a4b-it-vis560}
IMAGES=${IMAGES:-testdata/glm_ocr/invoices/inv01.png testdata/glm_ocr/invoices/inv02.png testdata/glm_ocr/table.png}
MAXTOK=${MAXTOK:-160} PORT=${PORT:-18519} SETTLE=${SETTLE:-20}
case "$MODEL$VISION" in *"/Volumes/"*|*"/srv/models"*) echo "the archive is not a bench surface (CLAUDE.md)" >&2; exit 2;; esac
[ -f testdata/glm_ocr/table.png ] || { echo "run from the repo root" >&2; exit 2; }
mkdir -p "$OUT"
rc=0
for arm in "$@"; do
  lab=${arm%%=*} rest=${arm#*=} bin=${rest%%:*} envs=""
  [[ $rest == *:* ]] && envs=${rest#*:}
  { echo "[$(date '+%F %T %Z')] arm $lab: $bin ($(cat "$bin.rev" 2>/dev/null || basename "$bin")) env=[$envs]"
    echo "  gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)  load: $(cut -d' ' -f1-3 /proc/loadavg)  mem avail: $(awk '/MemAvailable/{printf "%.1f GB", $2/1048576}' /proc/meminfo)"; } | tee -a "$OUT/provenance.txt"
  env $envs "$bin" --model "$MODEL" --vision "$VISION" -backend cuda -moe-cache-experts --ctx 16384 --addr 127.0.0.1:$PORT >"$OUT/$lab.serve.log" 2>&1 </dev/null &
  pid=$!
  up=0
  for _ in $(seq 1 900); do
    curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && { up=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  [ $up = 1 ] || { echo "!! $lab: serve did not come up; see $OUT/$lab.serve.log" | tee -a "$OUT/provenance.txt"; kill $pid 2>/dev/null; rc=1; continue; }
  grep -E "decode path|prefill path|vision tower|fit:|KV plan" "$OUT/$lab.serve.log" | cut -c1-220 | sed 's/^/  /' | tee -a "$OUT/provenance.txt"
  for img in $IMAGES; do
    python3 - "$OUT" "$lab" "$PORT" "$img" "$MAXTOK" <<'PY' || rc=1
import base64, json, os, sys, time, urllib.request
out, lab, port, img, maxtok = sys.argv[1:6]
b64 = base64.b64encode(open(img, 'rb').read()).decode()
body = {"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
    {"type": "text", "text": "Transcribe every field of this form as key: value lines."}]}],
    "max_tokens": int(maxtok), "temperature": 0, "logprobs": True, "top_logprobs": 3}
t = time.time()
req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
r = json.load(urllib.request.urlopen(req, timeout=3600))
wall = time.time() - t
c = r['choices'][0]
name = os.path.splitext(os.path.basename(img))[0]
open(f'{out}/{lab}-{name}.txt', 'w').write(c['message']['content'])
json.dump({"wall_s": wall, "usage": r.get("usage"), "logprobs": (c.get('logprobs') or {}).get('content') or []}, open(f'{out}/{lab}-{name}.json', 'w'))
print(f'  {lab} {name}: {wall:.1f}s, {(r.get("usage") or {}).get("prompt_tokens")} prompt / {(r.get("usage") or {}).get("completion_tokens")} completion tokens: {c["message"]["content"][:90]!r}')
PY
  done
  grep -E "vision: decoded" "$OUT/$lab.serve.log" | cut -c1-200 | sed 's/^/  /'
  kill $pid; wait $pid 2>/dev/null; sleep "$SETTLE"
done
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc

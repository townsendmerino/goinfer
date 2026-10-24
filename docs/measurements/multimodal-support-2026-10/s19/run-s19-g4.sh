#!/usr/bin/env bash
# S19's G-S19d (docs/tasks/task-multimodal-support-2026-10.md, "S19"): the per-form turn time, old binary (main: CPU prefill and
# upload) against new (resident span prefill), cells in the order A B B A, one fresh server per cell, served as Francis serves
# (-backend cuda -moe-cache-experts --ctx 16384, the 560-soft-token vision dir). A cell sends two forms in a row, as an OCR job
# does: inv01 (the server's first image) then inv02; each request streams, so time to first token and the whole turn are
# both read. Each cell records the 1-minute load and the free memory at its start; a cell started with load above 8.0 is VOID
# (CLAUDE.md "Run budget") and re-run in place, and the record lists it. Run it under the timing lock.
# Usage (from the repo root): run-s19-g4.sh <out dir> <old serve binary> <new serve binary>
set -uo pipefail
OUT=${1:?}; OLD=${2:?}; NEW=${3:?}
MODEL=${MODEL:-$HOME/models/gemma4-26b-int4.giw}
VISION=${VISION:-$HOME/models/gemma-4-26b-a4b-it-vis560}
MAXTOK=${MAXTOK:-160} PORT=${PORT:-18520} SETTLE=${SETTLE:-20}
case "$MODEL$VISION" in *"/Volumes/"*|*"/srv/models"*) echo "the archive is not a bench surface (CLAUDE.md)" >&2; exit 2;; esac
mkdir -p "$OUT"
cell() { # cell <n> <label> <binary>
  local n=$1 lab=$2 bin=$3 tries=0
  while :; do
    local la; la=$(cut -d' ' -f1 /proc/loadavg)
    local mem; mem=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
    local gpu; gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
    if awk "BEGIN{exit !($la > 8.0)}"; then
      echo "cell $n $lab VOID: load $la > 8.0 at start (mem avail $mem GB, gpu $gpu); re-run after 60 s" | tee -a "$OUT/cells.txt"
      tries=$((tries+1)); [ $tries -ge 30 ] && { echo "cell $n: gave up after 30 void starts" | tee -a "$OUT/cells.txt"; return 1; }
      sleep 60; continue
    fi
    echo "cell $n $lab start $(date '+%T') load $la mem avail $mem GB gpu $gpu" | tee -a "$OUT/cells.txt"
    break
  done
  "$bin" --model "$MODEL" --vision "$VISION" -backend cuda -moe-cache-experts --ctx 16384 --addr 127.0.0.1:$PORT >"$OUT/cell$n-$lab.serve.log" 2>&1 </dev/null &
  local pid=$! up=0
  for _ in $(seq 1 900); do
    curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && { up=1; break; }
    kill -0 $pid 2>/dev/null || break; sleep 1
  done
  [ $up = 1 ] || { echo "cell $n $lab: serve did not come up" | tee -a "$OUT/cells.txt"; kill $pid 2>/dev/null; return 1; }
  grep -E "decode path|prefill path|vision tower" "$OUT/cell$n-$lab.serve.log" | cut -c1-200 | sed 's/^/  /' | tee -a "$OUT/cells.txt"
  for img in testdata/glm_ocr/invoices/inv01.png testdata/glm_ocr/invoices/inv02.png; do
    python3 - "$PORT" "$img" "$MAXTOK" "$n" "$lab" <<'PY' | tee -a "$OUT/cells.txt"
import base64, json, os, sys, time, urllib.request
port, img, maxtok, n, lab = sys.argv[1:6]
b64 = base64.b64encode(open(img, 'rb').read()).decode()
body = {"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
    {"type": "text", "text": "Transcribe every field of this form as key: value lines."}]}],
    "max_tokens": int(maxtok), "temperature": 0, "stream": True}
t = time.time(); ttft = None; text = []
req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=3600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith('data:') or line == 'data: [DONE]':
            continue
        d = json.loads(line[5:])
        for c in d.get('choices', []):
            s = (c.get('delta') or {}).get('content')
            if s:
                if ttft is None:
                    ttft = time.time() - t
                text.append(s)
wall = time.time() - t
print(f"  cell {n} {lab} {os.path.basename(img)}: ttft {ttft:.1f}s turn {wall:.1f}s chars {len(''.join(text))}")
PY
  done
  grep -E "vision: decoded" "$OUT/cell$n-$lab.serve.log" | cut -c1-160 | sed 's/^/  /' | tee -a "$OUT/cells.txt"
  kill $pid; wait $pid 2>/dev/null; sleep "$SETTLE"
}
{ echo "old: $OLD"; echo "new: $NEW ($(cat "$NEW.rev" 2>/dev/null))"; echo "model: $MODEL  vision: $VISION"; echo "started: $(date '+%F %T %Z')"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"; } | tee "$OUT/provenance.txt"
rc=0
cell 1 old "$OLD" || rc=1
cell 2 new "$NEW" || rc=1
cell 3 new "$NEW" || rc=1
cell 4 old "$OLD" || rc=1
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc

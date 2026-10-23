#!/usr/bin/env bash
# S20's decode cells (docs/tasks/task-multimodal-support-2026-10.md, "S20"): Gemma 4 26B-A4B on the CUDA resident with
# -moe-cache-experts, one fresh server per cell, one long-reply turn per cell. A cell is
#   <label>|<image|text>|<extra serve flags>
# and an image cell also passes --vision. Each cell records the 1-minute load, free host memory and GPU memory at its
# start, the serve banner's plan lines (fit, KV plan, C′ cache), GPU memory after load and after the turn, the turn's
# time to first token, wall, completion tokens and decode rate ((tokens-1) / (wall - ttft)), and serve's per-turn
# expert-cache line. A cell started with load above 8.0 is VOID (CLAUDE.md "Run budget") and re-run in place; the record
# lists it. The reply text goes to <out>/cell<n>-<label>.reply.txt for the identity checks. Run it under the timing lock.
# Usage (from the repo root): run-s20-cells.sh <out dir> <serve binary> <cell>...
set -uo pipefail
OUT=${1:?}; BIN=${2:?}; shift 2
MODEL=${MODEL:-$HOME/models/gemma4-26b-int4.giw}
VISION=${VISION:-$HOME/models/gemma-4-26b-a4b-it-vis560}
IMG=${IMG:-testdata/glm_ocr/invoices/inv01.png}
MAXTOK=${MAXTOK:-1500} PORT=${PORT:-18530} SETTLE=${SETTLE:-20}
case "$MODEL$VISION" in *"/Volumes/"*|*"/srv/models"*) echo "the archive is not a bench surface (CLAUDE.md)" >&2; exit 2;; esac
mkdir -p "$OUT"
IMG_PROMPT="Transcribe every field of this form as key: value lines. Then, for each field, write two or three sentences on what it records, how it relates to the other fields, and what a clerk checking this document should verify about it."
TEXT_PROMPT="Write a detailed explanation, about 1,200 words, of how a supplier invoice moves through an accounts-payable department: receipt, data capture, three-way matching against the purchase order and goods receipt, approval routing, exception handling, payment scheduling and archiving. Use headings and give concrete examples of each check."
gpumem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
n=0
cell() { # cell <label> <image|text> <extra flags>
  local lab=$1 mode=$2 extra=$3 tries=0
  n=$((n+1))
  while :; do
    local la; la=$(cut -d' ' -f1 /proc/loadavg)
    local mem; mem=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
    if awk "BEGIN{exit !($la > 8.0)}"; then
      echo "cell $n $lab VOID: load $la > 8.0 at start (mem avail $mem GB, gpu $(gpumem) MiB); re-run after 60 s" | tee -a "$OUT/cells.txt"
      tries=$((tries+1)); [ $tries -ge 30 ] && { echo "cell $n: gave up after 30 void starts" | tee -a "$OUT/cells.txt"; return 1; }
      sleep 60; continue
    fi
    echo "cell $n $lab ($mode; $extra) start $(date '+%T') load $la mem avail $mem GB gpu $(gpumem) MiB" | tee -a "$OUT/cells.txt"
    break
  done
  local vis=(); [ "$mode" = image ] && vis=(--vision "$VISION")
  # shellcheck disable=SC2086
  "$BIN" --model "$MODEL" "${vis[@]}" -backend cuda -moe-cache-experts $extra --addr 127.0.0.1:$PORT >"$OUT/cell$n-$lab.serve.log" 2>&1 </dev/null &
  local pid=$! up=0
  for _ in $(seq 1 900); do
    curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && { up=1; break; }
    kill -0 $pid 2>/dev/null || break; sleep 1
  done
  [ $up = 1 ] || { echo "cell $n $lab: serve did not come up" | tee -a "$OUT/cells.txt"; tail -5 "$OUT/cell$n-$lab.serve.log" | sed 's/^/  /' | tee -a "$OUT/cells.txt"; kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep "$SETTLE"; return 1; }
  grep -E "C′ cache|fit:|KV plan|windowed KV|vision tower|drafter|decode path" "$OUT/cell$n-$lab.serve.log" | cut -c1-260 | sed 's/^/  /' | tee -a "$OUT/cells.txt"
  echo "  gpu after load: $(gpumem) MiB" | tee -a "$OUT/cells.txt"
  local prompt=$TEXT_PROMPT; [ "$mode" = image ] && prompt=$IMG_PROMPT
  python3 - "$PORT" "$mode" "$IMG" "$MAXTOK" "$n" "$lab" "$prompt" "$OUT/cell$n-$lab.reply.txt" <<'PY' | tee -a "$OUT/cells.txt"
import base64, json, sys, time, urllib.request
port, mode, img, maxtok, n, lab, prompt, replyf = sys.argv[1:9]
content = [{"type": "text", "text": prompt}]
if mode == "image":
    b64 = base64.b64encode(open(img, 'rb').read()).decode()
    content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}] + content
body = {"messages": [{"role": "user", "content": content}], "max_tokens": int(maxtok), "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True}}
t = time.time(); ttft = None; text = []; ntok = None
req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=3600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith('data:') or line == 'data: [DONE]':
            continue
        d = json.loads(line[5:])
        if d.get('usage'):
            ntok = d['usage'].get('completion_tokens')
        for c in d.get('choices', []):
            s = (c.get('delta') or {}).get('content')
            if s:
                if ttft is None:
                    ttft = time.time() - t
                text.append(s)
wall = time.time() - t
open(replyf, 'w').write(''.join(text))
rate = (ntok - 1) / (wall - ttft) if ntok and ttft is not None and wall > ttft else float('nan')
print(f"  cell {n} {lab}: ttft {ttft:.1f}s turn {wall:.1f}s tokens {ntok} decode {rate:.2f} tok/s chars {len(''.join(text))}")
PY
  echo "  gpu after turn: $(gpumem) MiB" | tee -a "$OUT/cells.txt"
  grep -E "expert cache:|vision: (encoded|decoded)" "$OUT/cell$n-$lab.serve.log" | cut -c1-200 | sed 's/^/  /' | tee -a "$OUT/cells.txt"
  kill $pid; wait $pid 2>/dev/null; sleep "$SETTLE"
}
{ echo "binary: $BIN ($(cat "$BIN.rev" 2>/dev/null))"; echo "model: $MODEL  vision: $VISION  image: $IMG  max_tokens: $MAXTOK"
  echo "started: $(date '+%F %T %Z')"; echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"; } | tee -a "$OUT/provenance.txt"
rc=0
for spec in "$@"; do
  IFS='|' read -r lab mode extra <<<"$spec"
  cell "$lab" "$mode" "$extra" || rc=1
done
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc

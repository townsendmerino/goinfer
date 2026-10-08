#!/usr/bin/env bash
# S9 on CUDA part B's speed record (docs/tasks/task-multimodal-support-2026-10.md, "S9 on CUDA, part B"): the real Gemma 4 E2B image turn on CUDA, time to the first
# token (max_tokens=1, non-streamed wall time of the request), prefill resident (NEW) against CPU prefill + upload (OLD, the pre-part-B build), with the SAME tower
# placement on both sides: tower on the CPU, and tower on CUDA. Kill rule (registered): if the resident-prefill turn is not faster than the CPU-prefill turn at the
# default image budget, the path is removed. No other bar.
#
# Design: ROUNDS rounds; per round the four arms (new:cpu old:cpu new:auto old:auto), the order alternating by round (reversed on even rounds); one fresh server per arm.
# Per arm REPS requests: the same table image with a different question each time (the prompt differs, so no prefix is reused; the tower may cache the features per
# image, so rep 1 includes the tower and reps 2.. are the warm tower: both reported, never pooled). Ratios are per round (old / new), rounds side by side.
#
# Usage (from the repo root): run-s9b-speed.sh [out dir]   Estimate: ~25 min.  BIN=<dir with serve-cuda-new, serve-cuda-old, rev, rev.old> (default ~/goinfer-bench/s9b)
# Queue:  python3 scripts/night.py add s9b-e2b-image-ttft --est 25 --by "nobara session, S9 on CUDA part B" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s9b-speed.sh
set -uo pipefail
BIN=${BIN:-$HOME/goinfer-bench/s9b}
OUT=${1:-$HOME/goinfer-logs/s9b/speed-$(date +%F-%H%M)}
MODEL=${MODEL:-$HOME/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf}
VISION=${VISION:-$HOME/models/gemma-4-E2B-unq}
ROUNDS=${ROUNDS:-3}; REPS=${REPS:-4}; PORT=${PORT:-18793}
IMG=${IMG:-testdata/glm_ocr/table.png}
for p in "$BIN/serve-cuda-new" "$BIN/serve-cuda-old" "$MODEL" "$VISION" "$IMG"; do [ -e "$p" ] || { echo "FATAL: $p is missing (run from the repo root)" >&2; exit 2; }; done
case "$MODEL" in /Volumes/*|/srv/models/*) echo "FATAL: $MODEL is on the archive (CLAUDE.md)" >&2; exit 2;; esac
mkdir -p "$OUT"
{ echo "bin: $BIN (new rev $(cat "$BIN/rev" 2>/dev/null || echo ?), old rev $(cat "$BIN/rev.old" 2>/dev/null || echo ?))"; echo "model: $MODEL  vision: $VISION"; echo "rounds=$ROUNDS reps=$REPS"
  echo "started: $(date '+%F %T %Z')"; echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load: $(cat /proc/loadavg)"; } | tee "$OUT/provenance.txt"
pid=
trap '[ -n "$pid" ] && kill "$pid" 2>/dev/null; wait 2>/dev/null' EXIT
for round in $(seq 1 "$ROUNDS"); do
  arms="new:cpu old:cpu new:auto old:auto"; [ $((round % 2)) -eq 0 ] && arms="old:auto new:auto old:cpu new:cpu"
  for arm in $arms; do
    bin=${arm%%:*}; vd=${arm##*:}; lab=$bin-$vd
    echo "[$(date +%T)] round $round arm $lab"
    "$BIN/serve-cuda-$bin" --model "$MODEL" --backend cuda -vision-device "$vd" --kv-sessions 1 -ctx 4096 --vision "$VISION" --addr 127.0.0.1:$PORT >"$OUT/serve-r$round-$lab.log" 2>&1 </dev/null &
    pid=$!
    for _ in $(seq 1 240); do
      curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && break
      kill -0 $pid 2>/dev/null || { echo "serve ($lab) exited; see $OUT/serve-r$round-$lab.log" >&2; exit 1; }
      sleep 1
    done
    python3 - "$OUT" "$round" "$lab" "$PORT" "$REPS" "$IMG" <<'PY'
import base64, json, sys, time, urllib.request
out, rnd, lab, port, reps, img = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
b64 = base64.b64encode(open(img, 'rb').read()).decode()
QS = ["What does this image show? Answer briefly.", "Describe this image in one sentence.", "What is the title of the table in this image?",
      "Which regions does this table list?", "What period does this table cover?", "How many columns does this table have?"]
ts = []
for k in range(reps):
    body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text", "text": QS[k % len(QS)]}]}], "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r: json.loads(r.read())
    ts.append(time.time() - t0)
print(f"   {lab}: " + " ".join(f"{t:.2f}" for t in ts))
json.dump({"times_s": ts}, open(f"{out}/r{rnd}-{lab}.json", "w"))
PY
    rc=$?; kill $pid 2>/dev/null; wait $pid 2>/dev/null; pid=
    [ $rc -eq 0 ] || { echo "!! client failed (round $round arm $lab)" >&2; exit 1; }
    if [ "$bin" = new ] && ! grep -q "(prefill resident)" "$OUT/serve-r$round-$lab.log"; then echo "!! the new arm never logged a resident image prefill: the comparison would be vacuous" >&2; exit 1; fi
  done
done
python3 - "$OUT" "$ROUNDS" <<'PY'
import json, statistics as st, sys
out, rounds = sys.argv[1], int(sys.argv[2])
verdict = "PASS"
for vd in ("cpu", "auto"):
    print(f"== tower placement {vd}: first rep (includes the tower) and median of the later reps (warm tower), old / new per round")
    for r in range(1, rounds + 1):
        n, o = json.load(open(f"{out}/r{r}-new-{vd}.json"))["times_s"], json.load(open(f"{out}/r{r}-old-{vd}.json"))["times_s"]
        print(f"   round {r}: first new {n[0]:.2f}s old {o[0]:.2f}s ({o[0]/n[0]:.2f}x) | later new {st.median(n[1:]):.2f}s old {st.median(o[1:]):.2f}s ({st.median(o[1:])/st.median(n[1:]):.2f}x)")
        if not st.median(n[1:]) < st.median(o[1:]): verdict = "KILL (resident-prefill turn not faster: the path is removed)"
print("kill rule:", verdict)
PY
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"

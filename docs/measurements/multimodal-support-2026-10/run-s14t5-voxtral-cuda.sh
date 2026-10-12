#!/usr/bin/env bash
# G-S14t5, Voxtral half (docs/tasks/task-multimodal-support-2026-10.md, S14.6 lever B): Voxtral Mini served with --backend cuda, so its audio tower runs on the device
# (cuda/audio_encoder.go): s14e4c's two clips as input_audio, float32 byte-equal to the SAME transformers reference s14e4c graded against, int4 (serve's default quant) within
# the word-edit count s14e4c read. Correctness record, not timed. Pinned in $BIN: serve-cuda (rev beside it) and the two python scripts; the reference and the 36 s clip are the
# ones the 2026-10-10 rerun saved in $REF. Checkpoint ~/models/voxtral-mini-3b-2507 (local NVMe), never the archive.
set -uo pipefail
BIN=${BIN:-$HOME/goinfer-bench/s14t5}
REF=${REF:-$HOME/goinfer-logs/s14e4c-2026-10-10-rerun}
OUT=${1:-$HOME/goinfer-logs/s14t5-voxtral-$(date +%F)}
M=$HOME/models/voxtral-mini-3b-2507
WAV=${WAV:-$HOME/mycode/goinfer/testdata/speech/librispeech-1272-128104-0000.wav}
PORT=18474
fatal() { echo "FATAL: $*" >&2; exit 2; }
for f in serve-cuda rev s14e4_request.py s14e4c_grade.py; do [ -e "$BIN/$f" ] || fatal "$BIN/$f is missing"; done
for f in "$M/tekken.json" "$WAV" "$REF/libri6x.wav" "$REF/ref/ref.log" "$REF/ref/libri.gen.json"; do [ -e "$f" ] || fatal "$f is missing"; done
case "$OUT$M" in *"/srv/models"*|*"/Volumes/"*) fatal "an archive path";; esac
curl -fs "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && fatal "something already serves on $PORT"
mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "binary: $BIN/serve-cuda (rev $(cat "$BIN/rev"))"; echo "started: $(date '+%F %T %Z')"; echo "load: $(cat /proc/loadavg)"; free -g | sed -n 2,3p; nvidia-smi --query-gpu=memory.used --format=csv,noheader; } | tee "$OUT/provenance.txt"
step() { local name=$1; shift; local t0=$SECONDS rc; echo "[$(date +%T)] START $name"; "$@" > "$OUT/$name.log" 2>&1; rc=$?; printf '%-18s rc=%-3s %5ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$SUM"; return $rc; }
serve_arm() { local arm=$1; shift
  "$BIN/serve-cuda" --model "$M" --backend cuda "$@" --addr 127.0.0.1:$PORT > "$OUT/serve-$arm.log" 2>&1 < /dev/null &
  local pid=$!; local up=0
  for _ in $(seq 1 900); do curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && { up=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 1; done
  if [ "$up" = 1 ]; then
    grep -E "Voxtral|decode path" "$OUT/serve-$arm.log" | cut -c1-220 | tee -a "$SUM"
    step "req-$arm-libri" python3 "$BIN/s14e4_request.py" "$WAV" "$OUT/reply-$arm-libri.json" $PORT
    step "req-$arm-libri6x" python3 "$BIN/s14e4_request.py" "$REF/libri6x.wav" "$OUT/reply-$arm-libri6x.json" $PORT
    grep -E "Voxtral audio tower|audio:" "$OUT/serve-$arm.log" | cut -c1-220 | tee -a "$SUM"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/vram after $arm: /" | tee -a "$SUM"
  else echo "!! the $arm server did not come up: see $OUT/serve-$arm.log" | tee -a "$SUM"; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3; }
for arm in ${ARMS:-f32 int4}; do
  case $arm in f32) serve_arm f32 --quant f32;; int4) serve_arm int4;; esac
done
python3 "$BIN/s14e4c_grade.py" "$REF/ref" "$OUT" 2>&1 | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
exit 0

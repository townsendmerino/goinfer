#!/usr/bin/env bash
# G-S14e4c (docs/tasks/task-multimodal-support-2026-10.md, "G-S14e4", registered before the code): RECORD ONLY. Voxtral Mini served on the CPU:
#   1. ref    transformers float32, language=None, on the 5.9 s clip and the six-times 36 s clip (scripts/pin_voxtral_real.py --language none)
#   2. f32    serve --quant f32: both clips as input_audio -> byte comparison with the reference
#   3. int4   serve at its DEFAULT quant (int4, int4 head): both clips -> word comparison with float32, control tokens, the first characters
#   4. grade  scripts/s14e4c_grade.py -> summary.txt
# Pinned in $BIN: serve-cpu (rev beside it) and the python scripts. Checkpoint ~/models/voxtral-mini-3b-2507 (local NVMe), never the archive. Correctness record, not timed.
# Estimate ~20 min (the reference ~6, f32 serve ~9, int4 ~5); queue at 40:
#   python3 scripts/night.py add s14e4c --est 40 --by "nobara session, S14 G-S14e4c" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash <pinned>/run-s14e4c-night.sh
# S14E4C_DRY=1 checks the preconditions and prints the plan.
set -uo pipefail
BIN=${BIN:-$HOME/goinfer-bench/s14e4}
OUT=${1:-$HOME/goinfer-logs/s14e4c-$(date +%F)}
M=$HOME/models/voxtral-mini-3b-2507
WAV=${WAV:-$BIN/testdata/speech/librispeech-1272-128104-0000.wav}
PY=$HOME/g4venv/bin/python
MC=$HOME/goinfer-bench/s14e/mc
PORT=18472
fatal() { echo "FATAL: $*" >&2; exit 2; }
for f in serve-cpu rev pin_voxtral_real.py s14e4_request.py s14e4c_grade.py; do [ -e "$BIN/$f" ] || fatal "$BIN/$f is missing"; done
for f in "$M/tekken.json" "$M/model-00001-of-00002.safetensors" "$WAV" "$PY"; do [ -e "$f" ] || fatal "$f is missing"; done
case "$OUT$M" in *"/srv/models"*|*"/Volumes/"*) fatal "an archive path";; esac
curl -fs "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && fatal "something already serves on $PORT"
mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "binaries: $BIN (rev $(cat "$BIN/rev"))"; echo "started: $(date '+%F %T %Z')"; echo "load: $(cat /proc/loadavg)"; free -g | sed -n 2p; } | tee "$OUT/provenance.txt"
[ -n "${S14E4C_DRY:-}" ] && { echo "DRY: preconditions hold; plan = reference, serve f32 (2 clips), serve int4 (2 clips), grade"; exit 0; }
step() { local name=$1; shift; local t0=$SECONDS rc; echo "[$(date +%T)] START $name"; "$@" > "$OUT/$name.log" 2>&1; rc=$?; printf '%-10s rc=%-3s %5ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$SUM"; return $rc; }
# 6x clip: six copies of the clip with half a second of silence between, as the reference builds it
python3 - "$WAV" "$OUT/libri6x.wav" <<'EOF'
import sys, wave
w = wave.open(sys.argv[1]); p = w.getparams(); d = w.readframes(w.getnframes()); gap = b"\0\0" * 8000
o = wave.open(sys.argv[2], "wb"); o.setparams(p); o.writeframes((d + gap) * 6); o.close()
EOF
refstep() { ( cd "$BIN" && env PYTHONPATH="$MC" "$PY" pin_voxtral_real.py --model "$M" --out "$OUT/ref" --language none ); } # the script opens testdata/speech/... relative to its directory
step ref refstep
cp "$OUT/ref.log" "$OUT/ref/ref.log" 2>/dev/null
serve_arm() { local arm=$1; shift
  "$BIN/serve-cpu" --model "$M" --backend cpu "$@" --addr 127.0.0.1:$PORT > "$OUT/serve-$arm.log" 2>&1 < /dev/null &
  local pid=$!; local up=0
  for _ in $(seq 1 900); do curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && { up=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 1; done
  if [ "$up" = 1 ]; then
    grep -E "Voxtral|decode path" "$OUT/serve-$arm.log" | cut -c1-200 | tee -a "$SUM"
    step "req-$arm-libri" python3 "$BIN/s14e4_request.py" "$WAV" "$OUT/reply-$arm-libri.json" $PORT
    step "req-$arm-libri6x" python3 "$BIN/s14e4_request.py" "$OUT/libri6x.wav" "$OUT/reply-$arm-libri6x.json" $PORT
  else echo "!! the $arm server did not come up: see $OUT/serve-$arm.log" | tee -a "$SUM"; fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3; }
serve_arm f32 --quant f32
serve_arm int4
python3 "$BIN/s14e4c_grade.py" "$OUT/ref" "$OUT" 2>&1 | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
exit 0

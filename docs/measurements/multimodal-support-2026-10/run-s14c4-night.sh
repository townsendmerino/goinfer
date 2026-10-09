#!/usr/bin/env bash
# G-S14c4 (docs/tasks/task-multimodal-support-2026-10.md, "G-S14c4", registered before this script): word error rate of Qwen3-ASR-0.6B at int4 against float32 on the 73 LibriSpeech dummy clips.
#   0. data     ~/goinfer-bench/librispeech-dummy (scripts/s14c4_data.py; skipped when refs.json is there)
#   1. R        transformers 5.15.0 float32 on the converted checkpoint (scripts/s14c4_hf.py)                          -> hf.json
#   2. F and I  goinfer's library path, native weights and serve's int4 defaults (decoder TestQwen3ASRWER_arms)         -> go.json
#   3. D        Gemma 4 E4B (QAT GGUF + its tower, CPU) through serve, "Transcribe this audio." (record only)           -> gemma.json
#   4. grade    scripts/s14c4_grade.py                                                                                  -> summary.txt
# Pinned in $BIN (main at the rev in $BIN/rev): decoder.test (realckpt), serve-cpu, the three python scripts. Checkpoints from ~/models (local NVMe), never the archive.
# Estimate ~55 min (R 7, F 6, I 3, encoder 1, D 20-25, loads). Queue:
#   python3 scripts/night.py add s14c4 --est 90 --by "nobara session, S14" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s14c4-night.sh
# S14C4_DRY=1 checks the preconditions and prints the plan. A step that fails is recorded and the later ones still run where they can.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s14c4}
OUT=${1:-$HOME/goinfer-logs/s14c4-$(date +%F)}
DATA=$HOME/goinfer-bench/librispeech-dummy
Q3=$HOME/models/qwen3-asr-0.6b
HFDIR=$HOME/goinfer-bench/qwen3asr-ref/hf
G4=$HOME/models/gemma-4-e4b-gguf/gemma-4-E4B_q4_0-it.gguf
G4V=$HOME/models/gemma-4-E4B-unq
PY=$HOME/g4venv/bin/python
fatal() { echo "FATAL: $*" >&2; exit 2; }
for f in decoder.test serve-cpu s14c4_hf.py s14c4_grade.py s14c4_gemma.py; do [ -e "$BIN/$f" ] || fatal "$BIN/$f is missing"; done
for f in "$Q3/model.safetensors" "$Q3/tokenizer_config.json" "$HFDIR/model.safetensors" "$DATA/refs.json" "$G4" "$G4V/model.safetensors"; do [ -e "$f" ] || fatal "$f is missing"; done
[ -x "$PY" ] || fatal "$PY is missing"
case "$OUT" in /srv/models*|/Volumes/*) fatal "$OUT is the archive";; esac
mkdir -p "$OUT"; SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "binaries: $BIN (main $(cat "$BIN/rev" 2>/dev/null))"; echo "started: $(date '+%F %T %Z')"; echo "load: $(cat /proc/loadavg)"; free -g | sed -n 2p; } | tee "$OUT/provenance.txt"
if [ -n "${S14C4_DRY:-}" ]; then echo "DRY: preconditions hold; plan = R, F+I, D, grade"; exit 0; fi
step() { local name=$1; shift; local t0=$SECONDS rc
  echo "[$(date +%T)] START $name"
  ( while sleep 60; do echo "[$(date +%T)] ... $name running $((SECONDS - t0))s; last: $(tail -n1 "$OUT/$name.log" | cut -c1-120)"; done ) &
  local tick=$!
  "$@" > "$OUT/$name.log" 2>&1; rc=$?
  kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
  printf '%-14s rc=%-3s %5ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$SUM"; return $rc; }
step hf timeout 60m "$PY" -I "$BIN/s14c4_hf.py" "$Q3" "$HFDIR" "$DATA" "$OUT/hf.json" || echo "!! the transformers arm failed" | tee -a "$SUM"
gotest() { ( cd "$SRC/decoder" && env GOINFER_HEAVY_TESTS=1 GOINFER_S14C4_DIR="$DATA" GOINFER_S14C4_OUT="$OUT/go.json" GOINFER_QWEN3ASR_DIR="$Q3" "$BIN/decoder.test" -test.run '^TestQwen3ASRWER_arms$' -test.v -test.timeout 90m ); }
step goinfer gotest || echo "!! the goinfer arms failed" | tee -a "$SUM"
# D: Gemma 4 E4B served on the CPU (the G-E4B-C2 flags: QAT GGUF as --model, the unquantized QAT safetensors as --vision, tower on the CPU, int8 head)
"$BIN/serve-cpu" --model "$G4" --vision "$G4V" --backend cpu -vision-device cpu --embed-int4=false --kv-sessions 1 -ctx 4096 --addr 127.0.0.1:18494 > "$OUT/gemma-serve.log" 2>&1 < /dev/null &
spid=$!; up=0
for _ in $(seq 1 900); do curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:18494/v1/models 2>/dev/null | grep -q 200 && { up=1; break; }; kill -0 $spid 2>/dev/null || break; sleep 1; done
if [ "$up" = 1 ]; then step gemma timeout 60m python3 "$BIN/s14c4_gemma.py" 18494 "$DATA" "$OUT/gemma.json" || echo "!! the Gemma arm failed (record only)" | tee -a "$SUM"
else echo "!! the Gemma server did not come up (record only): see $OUT/gemma-serve.log" | tee -a "$SUM"; fi
kill $spid 2>/dev/null; wait $spid 2>/dev/null
if [ -e "$OUT/hf.json" ] && [ -e "$OUT/go.json" ]; then
  python3 "$BIN/s14c4_grade.py" "$DATA/refs.json" "$OUT/hf.json" "$OUT/go.json" $( [ -e "$OUT/gemma.json" ] && echo "$OUT/gemma.json" ) 2>&1 | tee -a "$SUM"
else echo "!! not graded: hf.json or go.json is missing" | tee -a "$SUM"; fi
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
exit 0

#!/usr/bin/env bash
# Option D of the --embed-int4 decision (docs/tasks/task-multimodal-support-2026-10.md, "option D" and its amendment A1, registered before the harness code):
#   1. scripts/head_precision_hf.py: Hugging Face float32 logits on 32 fixed chat prompts x 32 greedy positions, seven models (the pinned copy in $BIN)
#   2. the positive control: decoder.test TestHeadPrecision_real on qwen2.5-0.5b-instruct with BOTH arms at the int8 pin (GOINFER_HP_CONTROL=1): must grade as exactly 0 / 0
#   3. the real run: every model, EmbedInt4=false against EmbedInt4=true, CPU backend, --quant int4 in both arms (HeadTable() asserted int8 against int4)
#   4. head_precision_grade.py: the per-model OK / COSTLY / MIXED readings and the registered map (the owner decides; this prints which pattern the numbers match)
# Pinned in $BIN (main at the rev in $BIN/rev): decoder.test (realckpt), head_precision_hf.py, head_precision_grade.py. Checkpoints from ~/models (local NVMe).
# Estimate ~110 min (HF ~45: the 7B and the 4B dominate; goinfer ~65). Queue:
#   python3 scripts/night.py add head-precision --est 150 --by "nobara session, embed-int4 D" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-head-precision-night.sh
# HP_DRY=1 checks the preconditions and prints the plan. A model whose HF side or goinfer load fails is recorded in its result and the others still run.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/hp}
OUT=${1:-$HOME/goinfer-logs/head-precision-$(date +%F)}
PY=$HOME/g4venv/bin/python
MODELS="qwen2.5-0.5b-instruct tinyllama-1.1b-chat qwen3-1.7b-bf16 qwen25vl-3b-instruct phi3-mini-4k gemma-3-4b-it olmo3-7b-think"
fatal() { echo "FATAL: $*" >&2; exit 2; }
for f in decoder.test head_precision_hf.py head_precision_grade.py; do [ -e "$BIN/$f" ] || fatal "$BIN/$f is missing"; done
[ -x "$PY" ] || fatal "$PY is missing"
for m in $MODELS; do [ -f "$HOME/models/$m/config.json" ] || fatal "$HOME/models/$m is missing"; done
case "$OUT" in /srv/models*|/Volumes/*) fatal "$OUT is the archive";; esac
free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9'); [ "$free_gb" -ge 15 ] || fatal "only ${free_gb} GB free, need ~10 for the logits"
mkdir -p "$OUT/hf" "$OUT/control"
SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "binaries: $BIN (main $(cat "$BIN/rev" 2>/dev/null))"; echo "started: $(date '+%F %T %Z')"; echo "free: ${free_gb} GB; load: $(cat /proc/loadavg)"; free -g | sed -n 2p; } | tee "$OUT/provenance.txt"
if [ -n "${HP_DRY:-}" ]; then echo "DRY: preconditions hold; plan = HF dump x7, control, real x7, grade"; exit 0; fi
step() { local name=$1; shift; local t0=$SECONDS rc
  echo "[$(date +%T)] START $name"
  ( while sleep 60; do echo "[$(date +%T)] ... $name running $((SECONDS - t0))s; last: $(tail -n1 "$OUT/$name.log" | cut -c1-120)"; done ) &
  local tick=$!
  "$@" > "$OUT/$name.log" 2>&1; rc=$?
  kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
  printf '%-18s rc=%-3s %5ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$SUM"; return $rc; }
gotest() { ( cd "$SRC/decoder" && env GOINFER_HEAVY_TESTS=1 GOINFER_HP_ROOT="$OUT/hf" "$@" "$BIN/decoder.test" -test.run '^TestHeadPrecision_real$' -test.v -test.timeout 150m ); }

step hf-dump "$PY" -I "$BIN/head_precision_hf.py" "$OUT/hf" $MODELS || echo "!! the HF dump returned non-zero; models that failed are recorded in their meta.json" | tee -a "$SUM"
grep -E "FAILED" "$OUT/hf-dump.log" | tee -a "$SUM"
step control gotest GOINFER_HP_ONLY=qwen2.5-0.5b-instruct GOINFER_HP_CONTROL=1 || echo "!! the control run failed" | tee -a "$SUM"
mkdir -p "$OUT/control/qwen2.5-0.5b-instruct" && cp "$OUT/hf/qwen2.5-0.5b-instruct/result.json" "$OUT/control/qwen2.5-0.5b-instruct/result.json" 2>/dev/null
"$PY" -I "$BIN/head_precision_grade.py" "$OUT/control" 2>&1 | tee -a "$SUM"
step real gotest || echo "!! the real run returned non-zero; per-model errors are in the result files" | tee -a "$SUM"
grep -E "^\[hp.*done|--- (PASS|FAIL)" "$OUT/real.log" | tail -20
"$PY" -I "$BIN/head_precision_grade.py" "$OUT/hf" 2>&1 | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
exit 0

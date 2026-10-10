#!/usr/bin/env bash
# G-S10q-c (docs/tasks/task-multimodal-support-2026-10.md, "S10, Qwen3-VL MoE", registered before its code): goinfer's CPU
# int8int8 and int4 arms on Qwen3-VL-30B-A3B against a layer-streaming Hugging Face float32 reference, on 8 image prompts
# (the four F2a images x two questions), the dense Qwen3-VL-2B as the bar's sibling. Step (b')'s design (run-g31b-night.sh).
#   0. controls   scripts/q3vlmoe_stream_controls.py on the tiny (streaming against ordinary, three planted defects red); a
#                 failed control aborts the job: no reading is made
#   1. 2B         goinfer's paths (its own int8int8 greedy continuations), the goinfer arms teacher-forced along them,
#                 and Hugging Face float32 ordinary
#   2. 30B        the same, the reference streamed layer by layer (58 GB read once)
#   3. grade      scripts/q3vlmoe_grade.py: agreement and KL per arm, the bootstrap, the registered rule
# Pinned in $BIN: decoder.test (realckpt, built from the branch at the rev in $BIN/rev), q3vlmoe_hf_stream.py,
# q3vlmoe_stream_controls.py, q3vlmoe_grade.py. The test binary runs from $SRC/decoder (it reads ../testdata).
# Checkpoints from ~/models (local NVMe), never the archive.
# Estimate ~90 min (30B loads ~15, its paths and arms ~40, the stream ~15, the 2B ~10, controls 1); queue at 150:
#   python3 scripts/night.py add s10q-c --est 150 --by "Claude, S10 Qwen3-VL MoE" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash <this script>
# S10Q_DRY=1 checks the preconditions and prints the plan. A step that fails is recorded and the later ones still run where
# they can; a missing output makes the grade refuse.
set -uo pipefail
SRC=${SRC:-$HOME/wt/goinfer-q3m}
BIN=${BIN:-$HOME/goinfer-bench/s10q-c}
OUT=${1:-$HOME/goinfer-logs/s10q-c-$(date +%F)}
PY=$HOME/.venv-vl/bin/python
DIR30=$HOME/models/qwen3-vl-30b-a3b-instruct
DIR2=$HOME/models/qwen3-vl-2b-instruct
fatal() { echo "FATAL: $*" >&2; exit 2; }
for f in decoder.test q3vlmoe_hf_stream.py q3vlmoe_stream_controls.py q3vlmoe_grade.py rev; do [ -e "$BIN/$f" ] || fatal "$BIN/$f is missing"; done
for f in "$DIR30/config.json" "$DIR2/config.json" "$SRC/testdata/qwen3vlmoe-tiny/model.safetensors" "$SRC/testdata/glm_ocr/table.png"; do [ -e "$f" ] || fatal "$f is missing"; done
[ -x "$PY" ] || fatal "$PY is missing"
for p in "$DIR30" "$DIR2" "$OUT"; do case "$p" in /srv/models*|/Volumes/*) fatal "$p is the archive (CLAUDE.md)";; esac; done
free_gb=$(df -BG --output=avail "$HOME/models" | tail -1 | tr -dc '0-9'); [ "$free_gb" -ge 40 ] || fatal "only ${free_gb} GB free under ~/models, need 40 (an int4 sidecar of the 30B is ~17 GB)"
mkdir -p "$OUT"
SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "binaries: $BIN (branch rev $(cat "$BIN/rev"))"; echo "started: $(date '+%F %T %Z')"; echo "free: ${free_gb} GB; load: $(cat /proc/loadavg)"; free -g | sed -n 2p; } | tee "$OUT/provenance.txt"
[ -n "${S10Q_DRY:-}" ] && { echo "DRY: preconditions hold; plan = controls, 2B (paths, goinfer arms, HF ordinary), 30B (paths, goinfer arms, HF stream), grade"; exit 0; }
step() { local name=$1; shift; local t0=$SECONDS rc
  echo "[$(date +%T)] START $name"
  ( while sleep 60; do echo "[$(date +%T)] ... $name running $((SECONDS - t0))s; last: $(tail -n1 "$OUT/$name.log" | cut -c1-120)"; done ) &
  local tick=$!
  "$@" > "$OUT/$name.log" 2>&1; rc=$?
  kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
  printf '%-18s rc=%-3s %5ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$SUM"; return $rc; }
gotest() { ( cd "$SRC/decoder" && env GOINFER_HEAVY_TESTS=1 "$@" "$BIN/decoder.test" -test.run '^TestQwen3VLMoe_goinferLogits$' -test.v -test.timeout 170m ); }
hf() { ( cd "$SRC" && "$PY" -I "$BIN/q3vlmoe_hf_stream.py" "$@" ); }

step controls "$PY" -I "$BIN/q3vlmoe_stream_controls.py" "$SRC/testdata/qwen3vlmoe-tiny" "$OUT/controls" || { echo "!! the controls did not hold: no reading is made" | tee -a "$SUM"; tail -8 "$OUT/controls.log" | tee -a "$SUM"; exit 1; }
grep "\[controls\]" "$OUT/controls.log" | tee -a "$SUM"

[ -e "$OUT/s2b.seqs.json" ] || step paths-2b gotest GOINFER_Q3M_DIR="$DIR2" GOINFER_Q3M_OUT="$OUT/s2b" GOINFER_Q3M_STEP=paths || echo "!! the 2B paths failed" | tee -a "$SUM"
[ -e "$OUT/s2b.int4.f32" ] || step goinfer-2b gotest GOINFER_Q3M_DIR="$DIR2" GOINFER_Q3M_OUT="$OUT/s2b" GOINFER_Q3M_STEP=logits || echo "!! the 2B goinfer arms failed" | tee -a "$SUM"
[ -e "$OUT/s2b.hf.f32" ] || step hf-2b hf "$DIR2" "$OUT/s2b.seqs.json" "$OUT/s2b.hf" ordinary || echo "!! the 2B Hugging Face reference failed" | tee -a "$SUM"

[ -e "$OUT/b30.seqs.json" ] || step paths-30b gotest GOINFER_Q3M_DIR="$DIR30" GOINFER_Q3M_OUT="$OUT/b30" GOINFER_Q3M_STEP=paths || echo "!! the 30B paths failed" | tee -a "$SUM"
[ -e "$OUT/b30.int4.f32" ] || step goinfer-30b gotest GOINFER_Q3M_DIR="$DIR30" GOINFER_Q3M_OUT="$OUT/b30" GOINFER_Q3M_STEP=logits || echo "!! the 30B goinfer arms failed" | tee -a "$SUM"
[ -e "$OUT/b30.hf.f32" ] || step hf-30b hf "$DIR30" "$OUT/b30.seqs.json" "$OUT/b30.hf" stream || echo "!! the 30B streaming reference failed" | tee -a "$SUM"

for f in b30.int8int8.f32 b30.int4.f32 b30.hf.f32 s2b.int8int8.f32 s2b.int4.f32 s2b.hf.f32; do [ -e "$OUT/$f" ] || { echo "!! $f is missing: no grade" | tee -a "$SUM"; exit 1; }; done
step grade "$PY" -I "$BIN/q3vlmoe_grade.py" "$OUT/b30" "$OUT/s2b" || echo "!! the grade failed" | tee -a "$SUM"
cat "$OUT/grade.log" | tee -a "$SUM"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
exit 0

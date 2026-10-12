#!/usr/bin/env bash
# G-S10q-e (docs/tasks/task-multimodal-support-2026-10.md, "S10, Qwen3-VL MoE", registered before this script): the paged CUDA
# path's teacher-forced logits on Qwen3-VL-30B-A3B at int4, graded against the Hugging Face float32 logits G-S10q-c already
# made, paired against the CPU int4 arm on the same positions.
#   1. cuda    TestQwen3VLMoe_cudaLogits: the CPU tower, the CPU prefill, UploadKV, then the resident's m-RoPE decode along the 8
#              pinned continuations; writes <out>.cuda.f32. A first load builds the cuda int4 sidecar in-process (about 90 s).
#   2. grade   scripts/q3vlmoe_cuda_grade.py: the paired difference and the registered rule; the verdict is its last line.
# Inputs from G-S10q-c, not re-made: b30.seqs.json, b30.hf.json/.f32, b30.int4.f32 (linked from $IN into the output dir).
# Pinned in $BIN: cuda.test (built with -tags 'cuda goinfer_testhooks' at the rev in $BIN/rev) and q3vlmoe_cuda_grade.py. The test
# runs from $SRC/cuda (it reads ../testdata/ images). Checkpoint from ~/models (local NVMe), never the archive. Correctness
# gate, not timed. Estimate 30 min; queue at 45:
#   python3 scripts/night.py add gs10q-e --est 45 --by "Claude, S10 Qwen3-VL MoE G-S10q-e" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash <this script>
# GS10QE_DRY=1 checks the preconditions and prints the plan. Exit 0 whatever the verdict (it is in summary.txt); nonzero only
# when the instrument itself did not produce a reading.
set -uo pipefail
SRC=${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
BIN=${BIN:-$HOME/goinfer-bench/gs10q-e}
IN=${IN:-$HOME/goinfer-logs/s10q-c-run}
OUT=${1:-$HOME/goinfer-logs/gs10q-e-$(date +%F)}
MODEL=$HOME/models/qwen3-vl-30b-a3b-instruct
fatal() { echo "FATAL: $*" >&2; exit 2; }
for f in cuda.test q3vlmoe_cuda_grade.py rev; do [ -e "$BIN/$f" ] || fatal "$BIN/$f is missing"; done
for f in b30.seqs.json b30.hf.json b30.hf.f32 b30.int4.f32; do [ -e "$IN/$f" ] || fatal "$IN/$f is missing"; done
for f in "$MODEL/config.json" "$MODEL/preprocessor_config.json" "$SRC/testdata/glm_ocr/table.png" "$SRC/testdata/gemma3_preprocess_image.png"; do
  [ -e "$f" ] || fatal "$f is missing"
done
for p in "$MODEL" "$OUT" "$IN"; do case "$p" in /srv/models*|/Volumes/*) fatal "$p is the archive (CLAUDE.md)";; esac; done
free_gb=$(df -BG --output=avail "$HOME/models" | tail -1 | tr -dc '0-9'); [ "$free_gb" -ge 30 ] || fatal "only ${free_gb} GB free under ~/models, need 30 (the cuda int4 sidecar is ~17 GB)"
mkdir -p "$OUT"
SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "binaries: $BIN (rev $(cat "$BIN/rev"))"; echo "started: $(date '+%F %T %Z')"; echo "free: ${free_gb} GB; load: $(cat /proc/loadavg)"; nvidia-smi --query-gpu=driver_version,memory.used --format=csv,noheader; } | tee "$OUT/provenance.txt"
[ -n "${GS10QE_DRY:-}" ] && { echo "DRY: preconditions hold; plan = cuda arm (30B paged int4) along the 8 pinned sequences, then grade"; exit 0; }
for f in b30.seqs.json b30.hf.json b30.hf.f32 b30.int4.f32; do ln -sf "$IN/$f" "$OUT/$f"; done
t0=$SECONDS
( while sleep 60; do echo "[$(date +%T)] ... cuda arm running $((SECONDS - t0))s; last: $(tail -n1 "$OUT/cuda.log" | cut -c1-140)"; done ) &
tick=$!
( cd "$SRC/cuda" && env GOINFER_HEAVY_TESTS=1 GOINFER_Q3M_DIR="$MODEL" GOINFER_Q3M_OUT="$OUT/b30" "$BIN/cuda.test" -test.run '^TestQwen3VLMoe_cudaLogits$' -test.v -test.timeout 100m ) > "$OUT/cuda.log" 2>&1
rc=$?
kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
printf '%-10s rc=%-3s %5ss\n' cuda "$rc" "$((SECONDS - t0))" | tee -a "$SUM"
[ "$rc" -eq 0 ] && [ -s "$OUT/b30.cuda.f32" ] || { echo "!! the CUDA arm produced no reading: no grade" | tee -a "$SUM"; tail -12 "$OUT/cuda.log" | tee -a "$SUM"; exit 1; }
python3 -I "$BIN/q3vlmoe_cuda_grade.py" "$OUT/b30" > "$OUT/grade.log" 2>&1 || { echo "!! the grader failed" | tee -a "$SUM"; tail -8 "$OUT/grade.log" | tee -a "$SUM"; exit 1; }
tee -a "$SUM" < "$OUT/grade.log"
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
exit 0

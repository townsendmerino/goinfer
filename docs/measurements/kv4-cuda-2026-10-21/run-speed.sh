#!/usr/bin/env bash
# KV4a's speed gate G2 (kv4-cuda-2026-10-21.md §2): decode at depth, new/old, A B B A per cell group (BENCH_ABBA=1), with G4's per-process
# memory sampler alongside. Binaries are pinned copies: old = merge base 8d7ca85b, new = the branch commit named in NEW_REV.
#   run-speed.sh <out dir>
set -euo pipefail
OUT=${1:?out dir}
mkdir -p "$OUT"
REPO=/home/francis/goinfer-bench/kv4-2026-10-21
BINS=/home/francis/goinfer-bench/kv4-bins
cd "$REPO"
python3 docs/measurements/kv4-cuda-2026-10-21/procmem_sampler.py "$OUT/procmem.jsonl" serve-cuda-old serve-cuda-new &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
export GOINFER_SERVE_CUDA=$BINS/serve-cuda-new GOINFER_SERVE_CUDA_OLD=$BINS/serve-cuda-old
export BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_ABBA=1 BENCH_REQUIRE_BACKEND=1 BENCH_SERVE_LOG_DIR="$OUT/serve-logs"
# bench_peer refuses to start on a box whose 1-minute load is over 1.0; wait for it (up to 15 minutes) before each invocation.
idle() { for _ in $(seq 90); do awk '{exit !($1 < 0.9)}' /proc/loadavg && return 0; sleep 10; done; echo "box not idle after 15 min"; return 1; }
PHASES=${PHASES:-qwen deep gemma}
for ph in $PHASES; do
  idle
  case $ph in
    qwen)  BENCH_MODELS=1.5B,7B BENCH_DEPTHS=2048,3900 python3 scripts/bench_peer.py "$OUT/decode-qwen.json" 2>&1 | tee "$OUT/decode-qwen.log" ;;
    # the 8,000-token cells need a context past the harness's 4,096 pin; both arms get the same explicit -ctx
    deep)  BENCH_CTX=12288 BENCH_MODELS=1.5B,7B BENCH_DEPTHS=8000 python3 scripts/bench_peer.py "$OUT/decode-deep.json" 2>&1 | tee "$OUT/decode-deep.log" ;;
    gemma) BENCH_MODELS=gemma3-1b BENCH_DEPTHS=2048,3900 python3 scripts/bench_peer.py "$OUT/decode-gemma3.json" 2>&1 | tee "$OUT/decode-gemma3.log" ;;
  esac
done
echo "RUN-SPEED-DONE"

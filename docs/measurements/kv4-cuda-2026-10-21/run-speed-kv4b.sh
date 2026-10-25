#!/usr/bin/env bash
# KV4b's speed check (kv4-cuda-2026-10-21.md §2, G5): decode at depth 128 and 2,048 on one checkpoint per f16 family group, the same
# binary at -kv f16 (new) against -kv f32 (old), A B B A per cell group (BENCH_ABBA=1). Bar: at least 0.98 at both depths. The binary is
# built here from the worktree the job runs in (W), so it is the tree the job was queued at.
#   env W=<pinned worktree> [MODELS="gemma3-4b cr7b ..."] bash <this script> <out dir>
# The script reports the ratios; the record grades them against the bar. A model whose harness run exits non-zero, refuses or leaves
# no results is VOID and fails the script. DeepSeek-V2-Lite is not in the default list: its CUDA batched prefill declines, so it runs on
# its own with BENCH_REQUIRE_BACKEND=0 (MODELS=dsv2lite).
set -uo pipefail
OUT=${1:?out dir}
W=${W:?pinned worktree}
mkdir -p "$OUT"
export PATH=$PATH:/usr/local/go/bin
BIN=$OUT/serve-cuda-kv4b
(cd "$W/cuda" && go build -tags cuda -o "$BIN" ./cmd/serve) || exit 1
git -C "$W" rev-parse HEAD > "$OUT/serve-cuda-kv4b.rev"
cd "$W"
export GOINFER_SERVE_CUDA=$BIN GOINFER_SERVE_CUDA_OLD=$BIN
# BENCH_REQUIRE_BACKEND=0 for a checkpoint whose batched prefill declines on CUDA (DeepSeek-V2-Lite): the scorer still voids a cell that
# does not decode on CUDA.
export BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_ABBA=1 BENCH_REQUIRE_BACKEND=${BENCH_REQUIRE_BACKEND:-1} BENCH_SERVE_LOG_DIR="$OUT/serve-logs"
export BENCH_GOINFER_ARGS="-kv f16" BENCH_GOINFER_OLD_ARGS="-kv f32"
# Directory checkpoints have no file size for the heavy-load gate to read: the resident host memory each needs, in GB.
export BENCH_FIT_NEED_GB="qmoe=16,cr7b=8,qvl3=6,dsv2lite=16,gemma3-4b=4,E2B=4"
# group 1 MoE + paged experts, 2 sliding window, 3 pairwise RoPE, 4 m-RoPE, 5 hybrid (Gemma 4), 6 MLA
# bench_peer refuses to start on a box whose 1-minute load is over 1.0; wait for it (up to 15 minutes) before each model.
idle() { for _ in $(seq 90); do awk '{exit !($1 < 0.9)}' /proc/loadavg && return 0; sleep 10; done; echo "box not idle after 15 min"; return 1; }
rc=0
for m in ${MODELS:-qmoe gemma3-4b cr7b qvl3 E2B}; do
  idle || { echo "VOID: $m not started (box busy)"; rc=1; continue; }
  BENCH_MODELS=$m BENCH_DEPTHS=128,2048 python3 scripts/bench_peer.py "$OUT/decode-$m.json" 2>&1 | tee "$OUT/decode-$m.log"
  hrc=${PIPESTATUS[0]}
  # bench_peer writes its results file before the cells run, so a crash leaves one behind: its exit code and log decide, not the file.
  if [ "$hrc" != 0 ] || [ ! -s "$OUT/decode-$m.json" ] || grep -qE "^Traceback|REFUSED" "$OUT/decode-$m.log"; then
    echo "VOID: $m (bench_peer exit $hrc; the harness failed, refused or left no results)"; rc=1
  fi
done
echo "RUN-SPEED-KV4B-DONE"
exit $rc

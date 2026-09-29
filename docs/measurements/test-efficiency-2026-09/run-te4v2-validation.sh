#!/bin/bash
# TE4-SEQ-v2 held-out validation corpus (te4-seq-v2-2026-09-28.md §2.2), on nobara, at night. Pre-registered in
# 5551e9c3 before any v2 code or data; the owner's 2026-09-28 decision (§3.3) makes this corpus the whole validation.
#
#   bash run-te4v2-validation.sh cuda    # gates 1-6: CUDA 0.5B A/A (Q 6); 1.5B, 7B A/A (Q 4); 0.5B/1.5B/7B vs Ollama (Q 4)
#   bash run-te4v2-validation.sh cpu     # gates 7-8: CPU 0.5B A/A (Q 4); CPU 1.5B vs Ollama (Q 4)
#
# Every gate runs to its cap (BENCH_ABBA quads, BENCH_RUNS=3, depth 128) and nothing stops early: the rule is replayed
# afterwards by te4v2_replay.py over the recorded blocks. Pinned: bench_peer.py and the replay come from a worktree at
# 25950af1, and both serve binaries were built from it (aikit v1.50.2, the go.mod pin). Night defaults apply
# (BENCH_MAX_LOADAVG=1.0). Runs under night.py, which holds the TE9 timing lock; bench_peer.py inherits it.
set -u
PART=${1:?usage: run-te4v2-validation.sh cuda|cpu}
B=$HOME/goinfer-bench/te4v2-2026-09-28
T=$B/tree
R=$B/results
mkdir -p "$R"
cd "$T" || exit 1
for f in "$B/serve-cuda-25950af1" "$B/serve-cpu-25950af1"; do [ -x "$f" ] || { echo "missing pinned binary $f"; exit 1; }; done
[ "$(git -C "$T" rev-parse HEAD)" = 25950af14b04a215aff333d9261784b11c7db457 ] || { echo "worktree is not at 25950af1"; exit 1; }
export GOINFER_SERVE_CUDA=$B/serve-cuda-25950af1 GOINFER_SERVE_CUDA_OLD=$B/serve-cuda-25950af1
export GOINFER_SERVE_CPU=$B/serve-cpu-25950af1 GOINFER_SERVE_CPU_OLD=$B/serve-cpu-25950af1
export BENCH_RUNS=3 BENCH_DEPTHS=none

gate() { # name quads backend models engines
  local out=$R/$1.json
  echo "=== $(date '+%F %T %Z') $(date +%s) START $1 (ABBA $2, $3, $4, $5)" | tee -a "$R/timeline.txt"
  BENCH_ABBA=$2 BENCH_BACKENDS=$3 BENCH_MODELS=$4 BENCH_ENGINES=$5 python3 scripts/bench_peer.py "$out"
  local rc=$?
  echo "=== $(date '+%F %T %Z') $(date +%s) END $1 rc=$rc" | tee -a "$R/timeline.txt"
}

case $PART in
cuda)
  gate g1-cuda-0.5B-aa 6 cuda 0.5B goinfer,goinfer_old
  gate g23-cuda-aa 4 cuda 1.5B,7B goinfer,goinfer_old
  gate g456-cuda-peer 4 cuda 0.5B,1.5B,7B goinfer,ollama
  ;;
cpu)
  gate g7-cpu-0.5B-aa 4 cpu 0.5B goinfer,goinfer_old
  gate g8-cpu-1.5B-peer 4 cpu 1.5B goinfer,ollama
  ;;
*) echo "unknown part $PART"; exit 2 ;;
esac

# The replay, over whatever gates exist so far (it names any that are missing). It is graded in the morning.
python3 -B docs/measurements/test-efficiency-2026-09/te4v2_replay.py replay "$R"/g*.json > "$R/replay-after-$PART.txt" 2>&1
echo "=== $(date '+%F %T %Z') DONE $PART (replay: $R/replay-after-$PART.txt)" | tee -a "$R/timeline.txt"

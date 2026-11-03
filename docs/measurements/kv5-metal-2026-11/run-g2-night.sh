#!/bin/bash
# KV5-2's G2 cells that need a quiet Mac (docs/tasks/task-kv-architecture-2026-10.md, KV5's plan): the 7B's decode cells, A B B A, and a
# repeat of gemma3-1b's two cells at depth, two quads each, as context for the kill rule. G4 as amended: any swap-out voids a cell, swap-ins
# are recorded, three voids in a row stop the run. Pinned binaries and harness: the tree may move tonight.
# new: a297113d (tuning round one); old: f8632013 (main just before KV5-1's merge).
# Exits 75 when the box refuses a cell for memory or load, so night.py retries, and each retry resumes from the cells already done.
D="$HOME/goinfer-work/kv5-2-g2"
cd "$D/scripts" || exit 1
for f in "$D/serve-new" "$D/serve-old" "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf" "$HOME/models/gemma3-1b-q4_k_m.gguf"; do
  [ -e "$f" ] || { echo "missing $f"; exit 1; }
done
mkdir -p "$D/out/serve-logs"
echo "new a297113d, old f8632013"
pmset -g batt | head -1
uptime
date '+start %H:%M'
pmset -g batt | head -1 | grep -q 'AC Power' || { echo "on battery: refused"; exit 75; }
export BENCH_DEPTH_BACKEND=metal BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=metal BENCH_REQUIRE_BACKEND=1
export BENCH_SERVE_LOG_DIR="$D/out/serve-logs" BENCH_LOAD_TIMEOUT=420 GOINFER_SERVE_METAL="$D/serve-new" GOINFER_SERVE_METAL_OLD="$D/serve-old"
export BENCH_SWAPOUT_VOID_MB=0 BENCH_SWAP_VOID_STOP=3 BENCH_IDLE_WAIT=180
unset BENCH_SWAPIN_VOID_MB
step() { # name env...
  name=$1; shift
  date "+== $name %H:%M:%S"
  env "$@" python3 -u bench_peer.py "$D/out/$name.json" > "$D/out/$name.log" 2>&1
  rc=$?
  date "+== $name rc=$rc %H:%M:%S"
  if [ "$rc" != 0 ]; then
    tail -1 "$D/out/$name.log" | cut -c1-200 | sed 's/VOID/void/g'
    if grep -q '^REFUSED' "$D/out/$name.log"; then exit 75; fi
    exit 1
  fi
}
step qwen7 BENCH_ABBA=1 BENCH_MODELS=7B BENCH_DEPTHS=2048,3900
step deep7 BENCH_ABBA=1 BENCH_MODELS=7B BENCH_PHASES=B BENCH_DEPTHS=8000 BENCH_DEEP_CTX=8192
step gemma-repeat BENCH_ABBA=2 BENCH_MODELS=gemma3-1b BENCH_PHASES=B BENCH_DEPTHS=2048,3900
date '+end %H:%M'

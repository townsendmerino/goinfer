#!/bin/bash
# KV5-6's G2 cells that need a quiet Mac (docs/tasks/task-kv-architecture-2026-10.md, KV5's plan: G2 and G4 whole at KV5-6): the 7B's
# decode cells at depths 128, 2,048, 3,900 and 8,000, A B B A, one quad a cell. G4 as amended: any swap-out voids a cell, swap-ins are
# recorded, three voids in a row stop the run. Pinned binaries and harness under ~/goinfer-work/kv5-6, so the tree may move.
# new: 0b2c9bd0 (KV5-6's product code with both review passes, main merged, built from an archive of the commit); old: f8632013 (main just before KV5-1's merge, KV5-2's serve-old).
# Exits 75 when the box refuses a cell for memory or load, so night.py retries, and each retry resumes from the cells already done.
D="$HOME/goinfer-work/kv5-6"
OLD="$HOME/goinfer-work/kv5-2-g2/serve-old"
cd "$D/new-0b2c9bd0/scripts" || exit 1
for f in "$D/serve-new" "$OLD" "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"; do
  [ -e "$f" ] || { echo "missing $f"; exit 1; }
done
mkdir -p "$D/g2-night/serve-logs"
echo "new 0b2c9bd0, old f8632013"
pmset -g batt | head -1
uptime
date '+start %H:%M'
pmset -g batt | head -1 | grep -q 'AC Power' || { echo "on battery: refused"; exit 75; }
export BENCH_DEPTH_BACKEND=metal BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=metal BENCH_REQUIRE_BACKEND=1
export BENCH_SERVE_LOG_DIR="$D/g2-night/serve-logs" BENCH_LOAD_TIMEOUT=420 GOINFER_SERVE_METAL="$D/serve-new" GOINFER_SERVE_METAL_OLD="$OLD"
export BENCH_SWAPOUT_VOID_MB=0 BENCH_SWAP_VOID_STOP=3 BENCH_IDLE_WAIT=180
unset BENCH_SWAPIN_VOID_MB
step() { # name env...
  name=$1; shift
  date "+== $name %H:%M:%S"
  env "$@" python3 -u bench_peer.py "$D/g2-night/$name.json" > "$D/g2-night/$name.log" 2>&1
  rc=$?
  date "+== $name rc=$rc %H:%M:%S"
  if [ "$rc" != 0 ]; then
    tail -1 "$D/g2-night/$name.log" | cut -c1-200 | sed 's/VOID/void/g'
    if grep -q '^REFUSED' "$D/g2-night/$name.log"; then exit 75; fi
    exit 1
  fi
}
step qwen7 BENCH_ABBA=1 BENCH_MODELS=7B BENCH_DEPTHS=2048,3900
step deep7 BENCH_ABBA=1 BENCH_MODELS=7B BENCH_PHASES=B BENCH_DEPTHS=8000 BENCH_DEEP_CTX=8192
date '+end %H:%M'

#!/bin/bash
# KV5-3's prefill TTFT read (docs/tasks/task-kv-architecture-2026-10.md, KV5's plan, G2's prefill cells; KV5-3 changes prefill's kernels):
# KV5-2's instrument, ttft_ab.py, unchanged, new serve against old, 1.5B then 7B, K = 512 and 3,900, three alternating pairs each.
# The harness is pinned: the scripts of a detached worktree at main 6ab62e5b, the old arm's revision. Run it under the exclusive lock:
#   python3 scripts/timing_lock.py run --wait 1800 --label kv5-3-ttft -- bash ~/goinfer-work/kv5-3/run-ttft.sh
set -u
B=$HOME/goinfer-bench/kv5-3
OUT=$HOME/goinfer-work/kv5-3
INSTR=$HOME/tmcode/goinfer/.claude/worktrees/op4m-c-2026-10-31/docs/measurements/kv5-metal-2026-11/ttft_ab.py
for f in "$B/serve-old" "$B/serve-new" "$B/wt-old/scripts/bench_peer.py" "$INSTR" "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"; do
  [ -e "$f" ] || { echo "missing $f"; exit 2; }
done
pmset -g batt | head -1 | grep -q "AC Power" || { echo "not on AC power: a timed cell on battery is void"; exit 3; }
cp "$INSTR" "$OUT/ttft_ab.py"
for MK in 1.5B 7B; do
  date "+$MK start %H:%M:%S"
  TTFT_MODEL=$MK TTFT_SCRIPTS="$B/wt-old/scripts" caffeinate -i python3 "$OUT/ttft_ab.py" "$B/serve-old" "$B/serve-new" "$OUT/ttft-$MK.json" 2>&1 | tee "$OUT/ttft-$MK.log"
  echo "$MK rc=${PIPESTATUS[0]}"
  date "+$MK end %H:%M:%S"
done

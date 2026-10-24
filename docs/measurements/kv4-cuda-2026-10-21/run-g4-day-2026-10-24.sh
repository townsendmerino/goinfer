#!/bin/bash
# KV4b G4's cells approved for 2026-10-24 by day, one after the other, each through run-g4.sh (which takes the timing lock for each
# phase): Qwen1.5-MoE K=8000 with four reference workers, then Command-R7B K=8000 with one (owner, 2026-10-23 4:45 pm PDT), then
# DeepSeek-V2-Lite K=3900 on its registered four prompts with four workers (owner, 5:52 pm). Starts at START
# (default 08:00 PDT on 2026-10-24) and needs no session alive. Its log goes where it is launched to write it.
#
#   setsid nohup env W=<pinned worktree> bash <this script> > <log> 2>&1 < /dev/null &
set -u
W=${W:?the pinned worktree}
START=${START:-2026-10-24 08:00}
now=$(date +%s); at=$(date -d "$START" +%s)
echo "tree: $W at $(git -C "$W" rev-parse HEAD); waiting until $START ($(( (at - now) / 60 )) min)"
[ "$at" -gt "$now" ] && sleep $(( at - now ))
rc=0
for cell in "QMOE 8000 4 4 3h" "CR7B 8000 1 4 3h" "DSV2L 3900 4 4 3h15m"; do
  set -- $cell
  echo "=== $1 K=$2, WORKERS=$3, N=$4, REF_TIMEOUT=$5, start $(date '+%F %T %Z'), load $(cut -d' ' -f1-3 /proc/loadavg)"
  env CELL="$1" DEPTH="$2" PHASE=both WORKERS="$3" N="$4" REF_TIMEOUT="$5" W="$W" bash "$W/docs/measurements/kv4-cuda-2026-10-21/run-g4.sh" || rc=1
done
echo "day runs finished $(date '+%F %T %Z') rc=$rc"
exit "$rc"

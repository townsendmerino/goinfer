#!/bin/bash
# G-I7-speed on the Mac (docs/tasks/task-integration-harness-2026-10.md, "I7", registered 2026-10-12): does a default
# context of 32768 slow a short turn? Dense decode at depth 128 on the 1.5B and the 7B under --backend metal: a trial
# build whose fit-by-default candidate is 32768 (branch m2b-candidate-32768) against main before M2 (4096), no -ctx on
# either, two arms, three counterbalanced quads of restarted cells (bench_peer.py phase A, BENCH_ABBA=3). Then one
# request of Claude Code's first-request size (20,199 tokens) to each model on the trial build, swap before and after.
# Runs with nobody watching: binaries, the harness, its prompts and the scripts are pinned copies in $B; night.py holds
# the timing lock and bench_peer.py inherits it.
#
#   B=~/goinfer-bench/m2b-2026-10-12 bash $B/run-g-i7-night.sh
set -u
B=${B:-$HOME/goinfer-bench/m2b-2026-10-12}
OUT=${OUT:-$B/out}
NEW=$B/bin/serve-metal-new
OLD=$B/bin/serve-metal-old
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
mkdir -p "$OUT" || exit 2
for f in "$NEW" "$OLD"; do [ -x "$f" ] || { echo "FATAL: $f is missing"; exit 2; }; done
for f in "$B/bench_peer.py" "$B/timing_lock.py" "$B/prompts.json" "$B/m2_served_check.py" "$B/grade_m2_speed.py" "$M15" "$M7"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }
done
{
  echo "started: $(date '+%F %T %Z')"; echo "host: $(uname -srm)"
  echo "new: $(cat "$NEW.rev" 2>/dev/null)"; echo "old: $(cat "$OLD.rev" 2>/dev/null)"
  echo "load: $(uptime)"; echo "swap: $(sysctl -n vm.swapusage)"; vm_stat | sed -n 1,8p
} > "$OUT/provenance.txt"
rc=0
cd "$B" || exit 2
BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=metal BENCH_MODELS=1.5B,7B BENCH_DEPTHS=none BENCH_ABBA=3 \
BENCH_REQUIRE_BACKEND=1 BENCH_PROMPTS=$B/prompts.json BENCH_SERVE_LOG_DIR=$OUT/serve-logs BENCH_LOAD_TIMEOUT=420 \
GOINFER_SERVE_METAL=$NEW GOINFER_SERVE_METAL_OLD=$OLD \
  python3 -u "$B/bench_peer.py" "$OUT/g-i7-speed.json" 2>&1 | tee "$OUT/g-i7-speed.log"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
python3 "$B/grade_m2_speed.py" "$OUT/g-i7-speed.json" "$OUT/serve-logs" 2>&1 | tee "$OUT/g-i7-speed.verdict.txt"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
for m in "$M15" "$M7"; do
  python3 -u "$B/m2_served_check.py" "$NEW" "$m" "$OUT/served" --tokens 20199 2>&1 | tee -a "$OUT/served-20k.log"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
done
echo "finished: $(date '+%F %T %Z') rc=$rc" >> "$OUT/provenance.txt"
exit "$rc"

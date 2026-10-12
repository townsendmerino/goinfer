#!/bin/bash
# M2 on the Mac, the night half (docs/tasks/task-minimax-code-compat-2026-09.md, "M2", registered 2026-10-12).
#   1. G-M2-speed: dense decode at depth 128 on the 1.5B and the 7B under --backend metal, the M2 build (no -ctx: the
#      fit-by-default context) against the build before it (no -ctx: 4096), two arms, three counterbalanced quads of
#      restarted cells (bench_peer.py phase A, BENCH_ABBA=3).
#   2. The served read the day could not make for the 7B (its resident build did not fit the daytime memory): no -ctx,
#      the banner's context and slots, one request of a coding agent's first-request size, swap before and after.
# Runs with nobody watching: both serve binaries, the harness, its prompts and this directory's scripts are pinned
# copies in $B; night.py holds the timing lock and bench_peer.py inherits it.
#
#   B=~/goinfer-bench/m2-2026-10-12 bash $B/run-g-m2-night.sh
set -u
B=${B:-$HOME/goinfer-bench/m2-2026-10-12}
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
  python3 -u "$B/bench_peer.py" "$OUT/g-m2-speed.json" 2>&1 | tee "$OUT/g-m2-speed.log"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
python3 "$B/grade_m2_speed.py" "$OUT/g-m2-speed.json" "$OUT/serve-logs" 2>&1 | tee "$OUT/g-m2-speed.verdict.txt"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
python3 -u "$B/m2_served_check.py" "$NEW" "$M7" "$OUT/served-7b" 2>&1 | tee "$OUT/served-7b.log"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
python3 -u "$B/m2_served_check.py" "$NEW" "$M15" "$OUT/served-1.5b" 2>&1 | tee "$OUT/served-1.5b.log"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
echo "finished: $(date '+%F %T %Z') rc=$rc" >> "$OUT/provenance.txt"
exit "$rc"

#!/usr/bin/env bash
# S1.0's 26B re-check, the Mac half (docs/tasks/task-multimodal-support-2026-10.md): Metal's per-position logits for
# the 26B (paged, guards on: the M26 rule) over three G3 prompts plus 16 greedy tokens, in two arms — the S1.0 fixes in
# ("fixed"), then GOINFER_S10_DROP=both teacher-forced over the same sequences. Not a timing run; it needs the night only
# because M26 by day is limited to interactive use. nobara supplies the CPU reference afterwards (the grading half).
# Each arm runs under swap_killwatch (+1 GB, 200 MB x2 tick rule), as every M26 run on this Mac does.
# Estimate: ~2 min per arm (smoke: 24 positions in 5 s after a 12 s load), queued at 15.
#
#   (cd metal && go test -c -tags goinfer_testhooks -o $BIN/metal.test .)   # BIN/rev names the commit
#   python3 scripts/night.py add s10-26b-dump --est 15 --by "Claude, S1.0" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-s10-26b-dump.sh
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s10-26b}
OUT=${1:-$HOME/goinfer-bench/s10-26b/run-$(date +%F)}
[ -x "$BIN/metal.test" ] || { echo "FATAL: $BIN/metal.test is missing"; exit 2; }
[ -e "$HOME/models/gemma4-26b-int4-v14st.metal.giw" ] || { echo "FATAL: the 26B .giw is missing from ~/models"; exit 2; }
mkdir -p "$OUT"
{ echo "binary:   $BIN/metal.test (rev $(cat "$BIN/rev" 2>/dev/null || echo unknown))"; echo "started:  $(date '+%F %T %Z')"
  sysctl vm.swapusage; df -h ~ | tail -1; } | tee "$OUT/provenance.txt"
cd "$SRC/metal" || exit 2
rc=0
arm() { # arm <name> [env...]
  local name=$1; shift
  echo "== $name: start $(date '+%T')"
  env GOINFER_HEAVY_TESTS=1 GOINFER_S10_DUMP_DIR="$OUT" "$@" "$BIN/metal.test" -test.run '^TestGemma4_26B_s10Dump$' -test.v -test.timeout 20m \
    > "$OUT/$name.log" 2>&1 &
  local pid=$!
  KILL_DELTA_MB=1024 TICK_MB=200 POLL_S=2 bash "$SRC/scripts/swap_killwatch.sh" "$pid" "$OUT/$name-killwatch.log" &
  local kw=$!
  wait "$pid"; local trc=$?
  wait "$kw" 2>/dev/null
  grep -E '^\[26B|^--- ' "$OUT/$name.log"
  if [ $trc -ne 0 ] || ! grep -q -- "--- PASS: TestGemma4_26B_s10Dump" "$OUT/$name.log"; then echo "!! $name did not PASS (rc $trc; see $OUT/$name.log)"; rc=1; fi
}
arm fixed
[ $rc -eq 0 ] && arm drop-both GOINFER_S10_DROP=both
ls -la "$OUT"
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc

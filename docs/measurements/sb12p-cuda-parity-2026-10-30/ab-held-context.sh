#!/usr/bin/env bash
# SB12p confirming A/B: TestKV2_scheduleIndependence_tiny + TestKV4_scrambledTables at 041884dbe, A = as the gate runs it,
# H = the same binary with GOINFER_VRAM_TRACE=1, whose sampler device holds the primary context open for the whole process.
# Order A H H A, under the caller's timing lock. Usage: ab-held-context.sh TESTBIN WORKTREE OUTDIR
set -u
BIN=${1:?testbin}; WT=${2:?worktree}; OUT=${3:?outdir}; mkdir -p "$OUT"
RUN='^(TestKV2_scheduleIndependence_tiny|TestKV4_scrambledTables)$'
i=0
for arm in A H H A; do
  i=$((i+1)); cd "$WT/cuda" || exit 2
  echo "== cell $i arm $arm start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg) memavail $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
  if [ $arm = H ]; then export GOINFER_VRAM_TRACE=1; else unset GOINFER_VRAM_TRACE; fi
  /usr/bin/time -f "wall %e s user %U sys %S" "$BIN" -test.run "$RUN" -test.count=1 -test.v -test.timeout 10m > "$OUT/held-$i-$arm.log" 2>&1
  echo "   rc=$? $(grep -E '^--- (PASS|FAIL|SKIP): Test[A-Za-z0-9_]+ ' "$OUT/held-$i-$arm.log" | tr '\n' ' ') $(tail -1 "$OUT/held-$i-$arm.log")"
  echo "   KV2 subtests: $(grep -E '^    --- PASS: TestKV2_scheduleIndependence_tiny/' "$OUT/held-$i-$arm.log" | sed -E 's/.*\/([^ ]+) \((.*)\)/\1=\2/' | tr '\n' ' ')"
done

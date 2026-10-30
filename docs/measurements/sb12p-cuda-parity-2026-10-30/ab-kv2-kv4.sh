#!/usr/bin/env bash
# SB12p A/B: TestKV2_scheduleIndependence_tiny (+ TestKV4_scrambledTables) at 6f638781d (fast 10-28 run) vs 041884dbe, A B B A.
set -u
T=$HOME/.claude/jobs/00b8812a/tmp
OUT=${1:?outdir}; mkdir -p "$OUT"
RUN='^(TestKV2_scheduleIndependence_tiny|TestKV4_scrambledTables)$'
i=0
for r in 6f638781d 041884dbe 041884dbe 6f638781d; do
  i=$((i+1))
  cd "$HOME/goinfer-bench/nobara-ci01-ab-$r/cuda" || exit 2
  echo "== cell $i rev $r start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg) memavail $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
  /usr/bin/time -f "wall %e s user %U sys %S maxrss %M KB" "$T/cuda-$r.test" -test.run "$RUN" -test.count=1 -test.v -test.timeout 10m \
     -test.cpuprofile "$OUT/cpu-$i-$r.prof" > "$OUT/cell-$i-$r.log" 2>&1
  echo "   rc=$? $(grep -E '^--- (PASS|FAIL|SKIP): Test[A-Za-z0-9_]+ ' "$OUT/cell-$i-$r.log" | tr '\n' ' ') $(tail -1 "$OUT/cell-$i-$r.log")"
done

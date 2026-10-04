#!/usr/bin/env bash
# H1.2 (docs/tasks/task-hardware-coverage-2026-10.md): the CPU parity suite and the tiny-fixture goldens under Intel SDE,
# so the AVX-512 VNNI paths nobody here owns (Ice Lake -icx, Sapphire Rapids -spr) execute, with Haswell -hsw as the
# control (AVX2 only, so it must match the native run on the 3700X).
#
# Runs with nobody watching: pinned pre-built test binaries (the tree may move before tonight), durable logs, no prompts.
# Not a timed measurement: emulation speed says nothing about speed.
#
# Provenance: decoder test binary built at the goinfer commit named in DBIN with aikit from the local checkout (v1.55.0-1-gbe7c35f, one
# commit past the v1.55.0 tag); linalg test binary from the same aikit checkout. SDE 10.13.1 (sde-external-10.13.1-2026-07-28).
set -uo pipefail   # no -e: one CPU model's failure must not stop the others

REPO=${REPO:-$HOME/mycode/goinfer}
SDE=${SDE:-$HOME/tools/sde/sde64}
DBIN=${DBIN:-$HOME/goinfer-logs/night/bin/decoder-sde-vnnigolden.test}
LBIN=${LBIN:-$HOME/goinfer-logs/night/bin/linalg-sde-v1.55.0.test}
OUT=${OUT:-$REPO/docs/measurements/sde-2026-10-04}
# Everything cheap under emulation. The LongPrompt tests take 4+ minutes EACH under SDE and are the same arithmetic the
# 15-fixture goldens already cover, so they are skipped and named as skipped.
RUN=${RUN:-'Golden|Parity|Int4|Int8|W4A8|Quant'}
SKIP=${SKIP:-'LongPrompt'}
CPUS=${CPUS:-"hsw icx spr"}
# linalg's FMA-peak test infers the clock from measured throughput (it read 0.35 GHz under SDE): a timing test, meaningless under emulation.
LSKIP=${LSKIP:-'FMAPeak'}
PER_RUN_MIN=${PER_RUN_MIN:-25}

mkdir -p "$OUT" || { echo "cannot create $OUT"; exit 2; }
for f in "$SDE" "$DBIN" "$LBIN"; do [ -x "$f" ] || { echo "missing $f"; exit 2; }; done
summary="$OUT/summary.txt"
: > "$summary"
echo "SDE goldens $(date -Is); decoder bin $DBIN; linalg bin $LBIN; run=$RUN skip=$SKIP" >> "$summary"
bad=0

for cpu in $CPUS; do
  # aikit's own linalg suite first (seconds): the kernel tests and SelfCheck on the emulated CPU.
  "$SDE" -"$cpu" -- "$LBIN" -test.skip "$LSKIP" -test.v -test.timeout "${PER_RUN_MIN}m" > "$OUT/linalg-$cpu.log" 2>&1
  lrc=$?
  # decoder tests read ../testdata, so they run from decoder/.
  ( cd "$REPO/decoder" && "$SDE" -"$cpu" -- "$DBIN" -test.run "$RUN" -test.skip "$SKIP" -test.v -test.timeout "${PER_RUN_MIN}m" > "$OUT/decoder-$cpu.log" 2>&1 )
  drc=$?
  for kind in linalg decoder; do
    log="$OUT/$kind-$cpu.log"
    if [ ! -s "$log" ]; then echo "$cpu $kind: NO LOG (did not run)" >> "$summary"; bad=1; continue; fi
    p=$(grep -c -- '^--- PASS' "$log"); s=$(grep -c -- '^--- SKIP' "$log"); f=$(grep -c -- '^--- FAIL' "$log")
    to=$(grep -c 'panic: test timed out' "$log")
    echo "$cpu $kind: pass=$p skip=$s fail=$f timeout=$to" >> "$summary"
    grep -- '^--- FAIL' "$log" | sed "s/^/    /" >> "$summary"
  done
  echo "$cpu exit codes: linalg=$lrc decoder=$drc" >> "$summary"
  # Any failure is unexpected. (TestDecodeParityInt4 has an "amd64-vnni" golden, captured under this same SDE, since 2026-10-04:
  # docs/measurements/sde-2026-10-04/README.md. A failure of it on icx/spr now means the VNNI path moved.)
  nfail=$(grep -h -- '^--- FAIL' "$OUT/decoder-$cpu.log" "$OUT/linalg-$cpu.log" 2>/dev/null | wc -l)
  [ "$nfail" -gt 0 ] && bad=1
  grep -q 'panic: test timed out' "$OUT/decoder-$cpu.log" && bad=1
done
echo "overall: $([ $bad = 0 ] && echo 'no unexpected failure' || echo 'UNEXPECTED FAILURE, TIMEOUT, OR MISSING LOG: read the per-CPU logs')" >> "$summary"
exit $bad

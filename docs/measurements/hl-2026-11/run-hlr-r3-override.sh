#!/usr/bin/env bash
# THE OVERRIDE COPY of run-hlr-r3.sh (Francis, 2026-11-08 03:12, "Force the Mac runs"): the one change is that the free-memory void is the model's
# file size, without the 2 GB margin; the load cap and the load's own fit guard stay. Its readings are not registered cells (README.md, "The 7B cells").
# HL-R3, HL-R's speed grade on Metal (docs/measurements/hl-2026-11/registration.md), registered before any HL-R code.
# TestHLRChainAB on the 1.5B and the 7B, one process per model and shape: greedy Generate under a gated processor with the chain
# running while the gate is closed (HL-R) against greedyChainOff (today's path), 9 reps of 256 tokens per arm, alternated.
# The graded lines are "HL-R3 METRIC hlr/today closed" (a) and "... toolcall" (b), on each model.
#
# Runs a test binary built at REV with -tags goinfer_testhooks, under the exclusive timing lock:
#   (cd backend/metal && go test -c -tags goinfer_testhooks -o ~/goinfer-bench/hl-2026-11/metal-tagged-<REV>.test .)
#   go build -o ~/goinfer-bench/hl-2026-11/hostmem ./cmd/hostmem
#   python3 scripts/timing_lock.py run --wait 600 --label hlr-r3 -- bash docs/measurements/hl-2026-11/run-hlr-r3-override.sh <REV> [all|7b]
# "7b" runs only the two 7B cells (logs in r3-7b), for a night run when the day's free memory was below the 7B's need.
#
# The registered void rule, checked at each cell's start and recorded in provenance.txt with the swap in use: the 1-minute load above
# 4.0 (half the cores), or reclaimable memory (cmd/hostmem -reclaimable, the reading the other timed runners use) below the model's
# file size plus 2 GB (the fit need plus the margin), voids the cell; it waits 60 s and starts again, up to 6 times. Exits non-zero
# unless every cell asked for ran and printed its METRIC line, and with 75 (the night queue re-queues it) when a cell never started
# because the guard stayed void.
set -uo pipefail
REV=${1:?usage: run-hlr-r3-override.sh <REV> [all|7b]}
WHICH=${2:-all}
BASE=$HOME/goinfer-bench/hl-2026-11
BIN=$BASE/metal-tagged-$REV.test
HOSTMEM=$BASE/hostmem
LOG=$HOME/goinfer-logs/hl-2026-11/r3
[ "$WHICH" = 7b ] && LOG=$HOME/goinfer-logs/hl-2026-11/r3-7b-override
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.v17.giw
rm -rf "$LOG"; mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
[ -x "$HOSTMEM" ] || { echo "missing $HOSTMEM"; exit 1; }
for f in "$M15" "$M7"; do [ -f "$f" ] || { echo "missing model $f"; exit 1; }; done
cd "$BASE"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
  echo "memory:   reclaimable $("$HOSTMEM" -reclaimable -mb) MiB; swap $(sysctl -n vm.swapusage)"
  echo "therm:    $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
  echo "OVERRIDE: Francis 2026-11-08 03:12 (\"Force the Mac runs\"): the registered free-memory void (file size + 2 GB) is relaxed to file size; readings are reported under this override, not as registered cells"
} | tee "$LOG/provenance.txt"

load1() { uptime | sed 's/.*load averages*: //' | awk '{gsub(",","",$1); print $1}'; }
ran=0; want=0; voided=0
grade() { # <log name> <model> <shape>
  local need_mb=$(( $(stat -f %z "$2") / 1048576 ))  # OVERRIDE (Francis, 2026-11-08 03:12, "Force the Mac runs"): the +2 GB headroom dropped; the load cap and the load's own fit guard stay
  for attempt in 1 2 3 4 5 6; do
    local l free swap
    l=$(load1); free=$("$HOSTMEM" -reclaimable -mb); swap=$(sysctl -n vm.swapusage)
    if awk -v l="$l" 'BEGIN{exit !(l > 4.0)}' || [ "${free:-0}" -lt "$need_mb" ]; then
      echo "== $1: VOID at start (attempt $attempt): load $l (cap 4.0), reclaimable ${free} MiB (need ${need_mb}); waiting 60 s" | tee -a "$LOG/provenance.txt"; sleep 60; continue
    fi
    echo "== $1 — $(date '+%T') load $l, reclaimable ${free} MiB (need ${need_mb}), swap: $swap" | tee -a "$LOG/provenance.txt"
    GOINFER_METAL_HLR_AB=1 GOINFER_METAL_HLR_MODEL="$2" GOINFER_METAL_HLR_SHAPE="$3" \
      "$BIN" -test.v -test.count=1 -test.timeout 30m -test.run '^TestHLRChainAB$' 2>&1 | tee "$LOG/$1.log"
    local rc=${PIPESTATUS[0]}
    echo "== $1 — finished $(date '+%T') rc=$rc, swap: $(sysctl -n vm.swapusage), reclaimable $("$HOSTMEM" -reclaimable -mb) MiB" | tee -a "$LOG/provenance.txt"
    if [ "$rc" -eq 0 ] && grep -q 'HL-R3 METRIC' "$LOG/$1.log"; then ran=$((ran + 1)); else echo "== $1: FAILED (rc $rc or no METRIC line)" | tee -a "$LOG/provenance.txt"; fi
    return
  done
  voided=$((voided + 1))
  echo "== $1: never started, the guard stayed void for 6 attempts" | tee -a "$LOG/provenance.txt"
}

if [ "$WHICH" = all ]; then
  want=$((want + 2))
  grade 1.5b-closed "$M15" closed
  grade 1.5b-toolcall "$M15" toolcall
fi
want=$((want + 2))
grade 7b-closed "$M7" closed
grade 7b-toolcall "$M7" toolcall

grep -h 'identity:\|METRIC\|loaded\|^--- \|VOID\|FAILED\|never started' "$LOG"/*.log "$LOG/provenance.txt" | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z'); $ran of $want cells counted" | tee -a "$LOG/provenance.txt"
[ "$ran" -eq "$want" ] && exit 0
[ "$voided" -gt 0 ] && exit 75
exit 1

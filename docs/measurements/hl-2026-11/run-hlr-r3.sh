#!/usr/bin/env bash
# HL-R3, HL-R's speed grade on Metal (docs/measurements/hl-2026-11/registration.md), registered before any HL-R code.
# TestHLRChainAB on the 1.5B and the 7B, one process per model and shape: greedy Generate under a gated processor with the chain
# running while the gate is closed (HL-R) against greedyChainOff (today's path), 9 reps of 256 tokens per arm, alternated.
# The graded lines are "HL-R3 METRIC hlr/today closed" (a) and "... toolcall" (b), on each model.
#
# Runs a test binary built at REV with -tags goinfer_testhooks, under the exclusive timing lock:
#   (cd backend/metal && go test -c -tags goinfer_testhooks -o ~/goinfer-bench/hl-2026-11/metal-tagged-<REV>.test .)
#   python3 scripts/timing_lock.py run --wait 600 --label hlr-r3 -- bash docs/measurements/hl-2026-11/run-hlr-r3.sh <REV>
# A cell started with the 1-minute load above 4.0 (half the cores) is void: it waits for the load to fall and re-runs in place.
set -uo pipefail
REV=${1:?usage: run-hlr-r3.sh <REV>}
BASE=$HOME/goinfer-bench/hl-2026-11
BIN=$BASE/metal-tagged-$REV.test
LOG=$HOME/goinfer-logs/hl-2026-11/r3
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.int4.metal.v17.giw
mkdir -p "$LOG"
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 1; }
for f in "$M15" "$M7"; do [ -f "$f" ] || { echo "missing model $f"; exit 1; }; done
cd "$BASE"
{
  echo "rev:      $REV"
  echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(uptime | sed 's/.*load averages*: //')"
  echo "therm:    $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
} | tee "$LOG/provenance.txt"

load1() { uptime | sed 's/.*load averages*: //' | awk '{gsub(",","",$1); print $1}'; }
grade() { # <log name> <model> <shape>
  for attempt in 1 2 3 4 5 6; do
    l=$(load1)
    if awk -v l="$l" 'BEGIN{exit !(l > 4.0)}'; then
      echo "== $1: load $l above 4.0 at the cell's start: VOID, waiting 60 s ($attempt)" | tee -a "$LOG/provenance.txt"; sleep 60; continue
    fi
    echo "== $1 — $(date '+%T') load $l" | tee -a "$LOG/provenance.txt"
    GOINFER_METAL_HLR_AB=1 GOINFER_METAL_HLR_MODEL="$2" GOINFER_METAL_HLR_SHAPE="$3" \
      "$BIN" -test.v -test.count=1 -test.timeout 30m -test.run '^TestHLRChainAB$' 2>&1 | tee "$LOG/$1.log" || true
    return
  done
  echo "== $1: never started, the load stayed above 4.0" | tee -a "$LOG/provenance.txt"
}

grade 1.5b-closed "$M15" closed
grade 1.5b-toolcall "$M15" toolcall
grade 7b-closed "$M7" closed
grade 7b-toolcall "$M7" toolcall

grep -h 'identity:\|METRIC\|loaded\|^--- \|VOID' "$LOG"/*.log "$LOG/provenance.txt" | tee "$LOG/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"

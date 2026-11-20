#!/bin/bash
# The registered sweep (README.md): five A B B A blocks at depth 128, then five at 3900, each block under its own exclusive
# hold of the timing lock so other sessions can take the card between blocks. BIN is the test binary built at the pinned commit.
set -u
BIN=$1; OUTDIR=$2
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$OUTDIR"
for depth in 128 3900; do
  for block in 1 2 3 4 5; do
    if [ "$(date +%H%M)" -ge 2130 ]; then echo "STOP: 21:30 PT reached before d$depth b$block"; exit 3; fi
    python3 "$REPO/scripts/timing_lock.py" run --wait 7200 --label "splitkv-phi3 d$depth b$block" -- \
      python3 "$REPO/docs/measurements/splitkv-phi3-2026-11-19/run_ab.py" --depth $depth --block $block \
        --bin "$BIN" --out "$OUTDIR/cells.tsv" --logdir "$OUTDIR/logs" || echo "block d$depth b$block exited $?"
  done
done
echo SWEEP-DONE

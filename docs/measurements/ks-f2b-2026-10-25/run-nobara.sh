#!/bin/bash
# KS-F2b on nobara, one family's two cells (README.md, "nobara"). A night.py job: night.py holds the timing lock. The arms are the
# test binaries the fit check built (A at 21245f4f4, B at 0058f0be9, in ~/goinfer-logs/ks-f2b-nobara-2026-10-25/bins).
set -uo pipefail
L=$HOME/goinfer-logs/ks-f2b-nobara-2026-10-25
S=$HOME/goinfer-bench/ks-f2b-nobara/docs/measurements/ks-f2b-2026-10-25/run_ksf2b_linux.py
case "${1:-}" in
  granite) M=granite/granite-4.0-h-tiny-Q8_0.gguf; NEED=13 ;;  # fit: VmHWM 12,635 MiB
  nemotron) M=nemotron-h-4b-instruct-128k; NEED=17 ;;          # fit: VmHWM 16,352 MiB
  *) echo "usage: run-nobara.sh granite|nemotron"; exit 2 ;;
esac
echo "== $(date '+%F %T %Z') KS-F2b $1: $M, need $NEED GiB; load $(cut -d' ' -f1-3 /proc/loadavg)"
sha256sum $L/bins/decoder-A.test $L/bins/decoder-B.test
rc=0
for d in 128 1024; do
  echo "== $(date '+%T') depth $d"
  python3 $S $L/bins $HOME/models/$M $d $NEED $L/readings-$1.jsonl || rc=1
done
echo "== $(date '+%F %T %Z') end, rc $rc"
exit $rc

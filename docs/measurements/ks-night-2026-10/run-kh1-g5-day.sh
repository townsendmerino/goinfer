#!/bin/bash
# KH1-G5 by day, 2026-10-24: run-kh1-g5.sh's two cold cells, serve built at 421a4347 and a570f8df with #203's hostmem reading
# applied (internal/hostmem, decoder/hostram_darwin.go; nothing the read measures), so the fit guard reads memory_pressure's figure.
# Exits 75 when the fit guard refuses a load.
set -u
B=$HOME/goinfer-bench/ks-day-2026-10-24
REPO=$HOME/goinfer-bench/ks-night-wt
OUT=$HOME/goinfer-logs/kh1-g5-day-$(date +%Y-%m-%d-%H%M)
cd "$REPO" || exit 1
(cd "$B/bin" && shasum -a 256 -c SHA256SUMS) || exit 1
echo "$(date '+%H:%M:%S') start: load $(sysctl -n vm.loadavg); swap $(sysctl -n vm.swapusage); out $OUT"
for rev in 421a4347 a570f8df; do
  python3 -u docs/measurements/mem0-2026-10/mem0_darwin.py --serve "$B/bin/serve-metal-$rev-hm203" --out "$OUT/$rev" --repo "$REPO" --cell metal:qwen35-9b
  rc=$?
  if grep -q 'NOT loaded' "$OUT/$rev/serve-metal-qwen35-9b.log" 2>/dev/null; then echo "FIT REFUSED at $rev (exit 75)"; exit 75; fi
  [ $rc -eq 0 ] || { echo "FAILED at $rev, rc $rc"; exit $rc; }
  echo "$(date '+%H:%M:%S') $rev done: swap $(sysctl -n vm.swapusage)"
done
echo "$(date '+%H:%M:%S') DONE: load $(sysctl -n vm.loadavg); swap $(sysctl -n vm.swapusage)"

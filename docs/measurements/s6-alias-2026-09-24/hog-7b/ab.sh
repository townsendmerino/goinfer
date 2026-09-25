#!/bin/bash
# S6 memory-hog arm: one server run per arm with a 6 GiB incompressible hog resident; alias first.
cd /Users/francistownsend-merino/tmcode/goinfer
MODEL="$1"; TAG="$2"; OUT=/tmp/hog/$TAG; mkdir -p $OUT; EXTRA="$3"
for i in 1 2; do
  if [ $i -eq 1 ]; then A=1; L=alias; else A=0; L=copy; fi
  echo "== $TAG $L $(date +%T) swap=$(sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+M')"
  python3 /tmp/hog/hog.py 6 > $OUT/hog-$L.log 2>&1 &
  H=$!
  for k in $(seq 1 120); do grep -q resident $OUT/hog-$L.log && break; sleep 1; done
  HR=$(ps -o rss= -p $H | awk '{printf "%.1f", $1/1048576}')
  echo "   hog RSS ${HR} GiB"
  if [ "$(echo "$HR > 6.5" | bc)" = 1 ]; then echo "   HOG TOO LARGE - aborting"; kill $H; exit 1; fi
  echo "   hog up $(date +%T), free $(memory_pressure | grep -oE '[0-9]+%') swap=$(sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+M')"
  ( for k in $(seq 1 240); do P=$(pgrep -f "^/tmp/serve-s6gate" | head -1); [ -n "$P" ] && { KILL_DELTA_MB=600 TICK_MB=80 TICK_N=2 POLL_S=1 exec bash scripts/swap_killwatch.sh "$P" $OUT/killwatch-$L.log; }; sleep 1; done ) > /dev/null 2>&1 &
  GOINFER_METAL_ALIAS=$A python3 scripts/moe_pager_ab.py --binary /tmp/serve-s6gate --model "$MODEL" --backend metal --no-stream-weights \
    --ctx 512 --port 18170 --label $L --kill-swap-mb 600 --extra "$EXTRA" --outdir $OUT > $OUT/$L.json 2> $OUT/$L.err
  echo "   exit=$? killwatch: $(tail -1 $OUT/killwatch-$L.log 2>/dev/null | cut -c1-100)"
  kill $H; wait $H 2>/dev/null
  if grep -q KILLING $OUT/killwatch-$L.log 2>/dev/null; then echo "   KILLED"; fi
  sleep 10
done
echo ALLDONE $(date +%T)

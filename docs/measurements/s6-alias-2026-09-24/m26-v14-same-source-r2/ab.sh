#!/bin/bash
# M26 (N=8) Metal, interleaved copy/alias/copy/alias — v14 re-run of m26-v13-shared, SAME safetensors source as v13 (same cell).
cd /Users/francistownsend-merino/tmcode/goinfer
OUT=/tmp/m26v14st-r2; MODEL=$HOME/models/gemma4-26b-int4-v14st.metal.giw
for i in 1 2 3 4; do
  if [ $((i%2)) -eq 1 ]; then A=1; L=alias$i; else A=0; L=copy$i; fi
  echo "== $L (GOINFER_METAL_ALIAS=$A) $(date +%T) swap=$(sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+M')"
  ( for k in $(seq 1 240); do
      P=$(pgrep -f "^/tmp/serve-s6gate" | head -1)
      [ -n "$P" ] && { KILL_DELTA_MB=600 TICK_MB=80 TICK_N=2 POLL_S=1 exec bash scripts/swap_killwatch.sh "$P" $OUT/killwatch-$L.log; }
      sleep 1
    done ) > /dev/null 2>&1 &
  GOINFER_METAL_ALIAS=$A python3 scripts/moe_pager_ab.py --binary /tmp/serve-s6gate --model $MODEL \
    --backend metal --no-stream-weights --ctx 512 --port 1811$i --label $L \
    --extra '-moe-cache-experts -moe-cache-slots 8' --kill-swap-mb 400 --outdir $OUT > $OUT/$L.json 2> $OUT/$L.err
  echo "exit=$? killwatch: $(tail -1 $OUT/killwatch-$L.log | cut -c1-100)"
  if grep -q "KILLING" $OUT/killwatch-$L.log 2>/dev/null; then echo "KILLED — stopping the A/B"; break; fi
  sleep 8
done
echo ALLDONE $(date +%T)

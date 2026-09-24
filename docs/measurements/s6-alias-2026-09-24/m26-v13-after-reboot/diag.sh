#!/bin/bash
cd /Users/francistownsend-merino/tmcode/goinfer
OUT=/tmp/m26s6/diag; mkdir -p $OUT; MODEL=$HOME/models/gemma4-26b-int4-v13.giw
echo "swap before: $(sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+M')  $(date +%T)"
( for k in $(seq 1 240); do
    P=$(pgrep -f "^/tmp/goinfer-metal-serve-s6" | head -1)
    if [ -n "$P" ]; then
      python3 - "$P" > $OUT/liveness.log <<'PY' &
import os,sys,time
pid=int(sys.argv[1]); t0=time.time()
while True:
    try: os.kill(pid,0)
    except ProcessLookupError:
        print("server pid %d GONE at wall %.3f (t+%.2fs from first sight)"%(pid,time.time(),time.time()-t0),flush=True); break
    time.sleep(0.05)
PY
      KILL_DELTA_MB=600 TICK_MB=80 TICK_N=2 POLL_S=1 exec bash scripts/swap_killwatch.sh "$P" $OUT/killwatch.log
    fi
    sleep 1
  done ) > /dev/null 2>&1 &
GOINFER_METAL_ALIAS=1 python3 scripts/moe_pager_ab.py --binary /tmp/goinfer-metal-serve-s6 --model $MODEL \
  --backend metal --no-stream-weights --ctx 512 --port 18120 --label diag \
  --extra '-moe-cache-experts -moe-cache-slots 8' --kill-swap-mb 400 --outdir $OUT > $OUT/result.json 2> $OUT/harness.err
echo "harness exit=$? $(date +%T)"; python3 -c "
import json;d=json.load(open('$OUT/result.json'));print({k:d.get(k) for k in ('load_s','tokens','request_error','server_exit_before_finish','ended_reason')})"
cat $OUT/liveness.log; tail -2 $OUT/killwatch.log | cut -c1-140
echo "swap after: $(sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+M')"

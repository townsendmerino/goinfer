#!/bin/bash
# By-day fit check for G-ids-3: start the pinned serve with the job's flags, read its banner, stop it.
B=/home/francis/goinfer-bench/ids3-2026-10
for arm in on off; do
  extra=""; [ $arm = off ] && extra="--carry-ids=false"
  log=$B/fitcheck-$arm.log
  $B/bin/serve-cuda-ids3 --model /home/francis/models/mellum2.1-thinking --backend cuda --ctx 16384 --addr 127.0.0.1:18090 --log-requests --moe-cache-experts $extra --log-token-ids $B/fitcheck-token-ids.jsonl > $log 2>&1 < /dev/null &
  pid=$!
  for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:18090/v1/models > /dev/null && break
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> $log
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> $log
  kill $pid; wait $pid
  echo "== $arm"; grep -E "C′|fit:|KV plan|context:|refus|error|windowed|ring|MiB" $log
done

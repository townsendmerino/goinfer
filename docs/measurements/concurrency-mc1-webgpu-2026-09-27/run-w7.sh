#!/bin/bash
# MC1 on WebGPU — W7 grading (docs/tasks/task-concurrency-2026-09.md "MC1 on WebGPU", pre-registered before any timing):
# the MacBook (M1 Pro 16 GB), WebGPU through wgpu-native's Metal backend, qwen2.5-coder-1.5b-instruct q4_k_m int4 from
# ~/models, serve defaults (-kv-sessions 4), greedy, 6 turns x 128 tokens, --fixed-nonce, a fresh server per cell.
# old = serve-webgpu @ $OLD (one KV slot); new = serve-webgpu @ $NEW (MC1-WebGPU).
set -u
OLD=${OLD:?}; NEW=${NEW:?}
cd "$HOME/tmcode/goinfer" || exit 1
B=$HOME/goinfer-bench/mc1-webgpu-2026-09-27
OUT=$B/w7.json
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
ts() { date '+%H:%M:%S'; }
# Idle gate: load1 <= 2.0 and no other serve-webgpu process (a predecessor's device memory is host RAM here).
gate() {
  waited=0
  while :; do
    l1=$(sysctl -n vm.loadavg | awk '{print $2}')
    procs=$(pgrep -f "serve-webgpu-" | wc -l | tr -d ' ')
    if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}' && [ "$procs" -eq 0 ]; then
      echo "$(ts) idle: load1=$l1 serve_procs=$procs"; return 0
    fi
    [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s (load1=$l1 serve_procs=$procs) — stopping"; exit 1; }
    [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 serve_procs=$procs (${waited}s)"
    sleep 5; waited=$((waited + 5))
  done
}
cell() {
  k=$1; n=$2
  case $k in old*) bin=$B/serve-webgpu-$OLD;; new*) bin=$B/serve-webgpu-$NEW;; esac
  gate
  echo "$(ts) cell $k clients=$n bin=$(basename "$bin")"
  BENCH_W7_MODEL=$M15 GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_w7_plain.py "$OUT" --clients "$n" --engines goinfer \
    --backend webgpu --key "$k" --fixed-nonce --server-log "$B/w7-servers.log" || { echo "$(ts) cell $k FAILED"; exit 1; }
  [ -s "$OUT" ] || { echo "$(ts) no results file after cell $k"; exit 1; }
}
case "$M15" in /srv/models/*|/Volumes/*) echo "model $M15 is on the archive, not the bench set"; exit 1;; esac
[ -f "$M15" ] || { echo "no model at $M15"; exit 1; }
for h in "$OLD" "$NEW"; do [ -x "$B/serve-webgpu-$h" ] || { echo "no binary $B/serve-webgpu-$h"; exit 1; }; done
echo "$(ts) == MC1-WebGPU W7 start; tree $(git rev-parse --short HEAD); old $OLD new $NEW; $(sw_vers -productVersion); load $(sysctl -n vm.loadavg)"
for n in 1 2 4; do
  for k in old${n}_1 new${n}_1 new${n}_2 old${n}_2 old${n}_3 new${n}_3; do cell "$k" "$n"; done
  echo "$(ts) == $n-client cells done"
done
echo "$(ts) == DONE"

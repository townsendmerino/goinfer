#!/bin/bash
# MC4 candidate "speculate when alone, batch under load" (-spec-adaptive), graded on CUDA. Pre-registered in
# docs/measurements/mc4-candidate-cuda-2026-10-01.md BEFORE this ran; gates in gates.py (same directory).
# nobara-pc, RTX 2070 SUPER, driver 595.91.07, ONE binary (serve-cuda @ efaae8a6, built once, named by its hash), three
# arms, a fresh server per cell:
#   b = batch       serve's defaults (MC3 on) -- the do-nothing arm
#   s = spec-excl   -spec ngram (today's: one generation at a time)
#   c = candidate   -spec ngram -spec-adaptive
set -u
cd /home/francis/mycode/goinfer || exit 1
B=/home/francis/goinfer-bench/mc4-candidate-cuda-2026-10-01
OUT=/home/francis/mycode/goinfer/docs/measurements/mc4-candidate-cuda-2026-10-01
BIN=$B/serve-cuda-efaae8a6
export GOINFER_SERVE_CPU=$BIN
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
ts() { date '+%H:%M:%S'; }
mkdir -p "$OUT/raw"
BASE_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
BASE_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
gate() {   # CUDA idle gate (spec-vs-batching-cuda-2026-09-28/run.sh's): load1 <= 2.0, GPU procs and memory at the run's baseline
  waited=0
  while :; do
    l1=$(awk '{print $1}' /proc/loadavg)
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    if awk -v l="$l1" 'BEGIN{exit !(l <= 2.0)}' && [ "$procs" -le "$BASE_PROCS" ] && [ "$used" -le $((BASE_MIB + 256)) ]; then
      echo "$(ts) idle: load1=$l1 gpu_used=${used}MiB compute_procs=$procs"; return 0
    fi
    [ "$waited" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s (load1=$l1 gpu_used=${used}MiB procs=$procs) -- stopping"; exit 1; }
    [ $((waited % 60)) -eq 0 ] && echo "$(ts) waiting for idle: load1=$l1 gpu_used=${used}MiB procs=$procs (${waited}s)"
    sleep 5; waited=$((waited + 5))
  done
}
# arm letter -> the serve args array
sargs() { case $1 in s) SA=("--serve-args=-spec=ngram");; c) SA=("--serve-args=-spec=ngram -spec-adaptive");; *) SA=();; esac; }
copy() { model=$1 out=$2 arm=$3 r=$4 n=$5; sargs $arm; gate; echo "$(ts) copy $out $arm$n _$r clients=$n"
  BENCH_W7_MODEL=$model python3 -u scripts/bench_spec_copy.py "$OUT/raw/$out" --key "$arm${n}_$r" --clients $n --backend cuda \
    "${SA[@]}" --server-log "$OUT/raw/servers.log"; }
stag() { arm=$1 r=$2; sargs $arm; gate; echo "$(ts) stagger ${arm}S_$r"   # rounds 12,9,6,3, a client joining every 2 s
  BENCH_W7_MODEL=$M15 python3 -u scripts/bench_spec_copy.py "$OUT/raw/stag-15b.json" --key "${arm}S_$r" --rounds-list 12,9,6,3 \
    --stagger-s 2 --backend cuda "${SA[@]}" --server-log "$OUT/raw/servers.log"; }
chat() { arm=$1 r=$2 n=$3; sargs $arm; gate; echo "$(ts) chat $arm${n}_$r clients=$n"
  BENCH_W7_MODEL=$M15 python3 -u scripts/bench_w7_plain.py "$OUT/raw/chat-15b.json" --clients $n --engines goinfer --backend cuda \
    --key "$arm${n}_$r" --fixed-nonce "${SA[@]}" --server-log "$OUT/raw/servers.log"; }
# three interleaved rounds, the arm order rotated each round (a Latin square, so no arm is always first or last)
order() { case $1 in 1) echo b s c;; 2) echo s c b;; 3) echo c b s;; esac; }
for m in "$M15" "$M7"; do case "$m" in /srv/models/*|/Volumes/*) echo "$(ts) $m is on the archive, not the bench set"; exit 1;; esac; [ -f "$m" ] || { echo "$(ts) no model at $m"; exit 1; }; done
[ -x "$BIN" ] || { echo "$(ts) no binary $BIN"; exit 1; }
echo "$(ts) == mc4-candidate-cuda start; tree $(git rev-parse --short HEAD); bin $(basename $BIN); driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader); gpu baseline ${BASE_MIB}MiB, ${BASE_PROCS} compute proc(s); $(cat /proc/loadavg)"
for r in 1 2 3; do for a in $(order $r); do copy "$M15" copy-15b.json $a $r 4; done; done
for r in 1 2 3; do for a in $(order $r); do copy "$M15" copy-15b.json $a $r 1; done; done
for r in 1 2 3; do for a in $(order $r); do chat $a $r 4; done; done
for r in 1 2 3; do for a in $(order $r); do chat $a $r 1; done; done
for r in 1 2 3; do for a in $(order $r); do stag $a $r; done; done
for r in 1 2 3; do for a in $(order $r); do copy "$M7" copy-7b.json $a $r 4; done; done
for r in 1 2 3; do for a in $(order $r); do copy "$M7" copy-7b.json $a $r 1; done; done
echo "$(ts) == DONE"
echo "grade: python3 docs/measurements/mc4-candidate-cuda-2026-10-01/gates.py"

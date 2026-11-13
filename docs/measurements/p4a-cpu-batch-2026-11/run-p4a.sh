#!/bin/bash
# P4a's speed check (docs/tasks/task-kv-precision-everywhere-2026-10.md, P4a; registration.md beside this file): nobara CPU, the 7B
# (qwen2.5-7b-instruct q4_k_m, over cpuBatchAutoMinBytes, so -cpu-batch auto engages the batcher), serve defaults (-max-concurrent 4,
# -kv-sessions 4, -cpu-batch auto), greedy, bench_w7_plain.py's 6 turns at --max-tokens 32, --fixed-nonce, a fresh server per cell.
# A = serve-cpu at OLD (main: an f16 or i8 cache's tokens decode per sequence), B = serve-cpu at NEW (#874: they batch).
# Cells: -kv f16 and -kv i8 at 4 clients, 10 pairs each (decisive); at 2 clients, 3 pairs each (reported). A B B A order.
#   run-p4a.sh <OLD sha> <NEW sha>      (run under the exclusive lock: python3 scripts/timing_lock.py run --label p4a -- bash <this> ...)
set -u
HERE=$(cd "$(dirname "$(realpath "$0")")" && pwd)
OLD=${1:?old sha}; NEW=${2:?new sha}
SRC=$HOME/mycode/goinfer; B=$HOME/goinfer-bench/p4a-cpu-batch; OUT=$HOME/goinfer-logs/p4a-cpu-batch-$(date +%F)
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
export PATH=/usr/local/go/bin:$PATH
case "$M7 $B $OUT" in *"/srv/models"*|*"/Volumes/"*) echo "FATAL: an archive path"; exit 2;; esac
[ -f "$M7" ] || { echo "FATAL: no $M7"; exit 2; }
mkdir -p "$B" "$OUT"
ts() { date '+%H:%M:%S'; }
for rev in $OLD $NEW; do # one binary per arm, each from its own detached worktree
  bin=$B/serve-cpu-$rev
  [ -x "$bin" ] && continue
  wt=$B/wt-$rev
  [ -d "$wt" ] || git -C "$SRC" worktree add -q --detach "$wt" "$rev" || { echo "FATAL: cannot check out $rev"; exit 2; }
  (cd "$wt" && CGO_ENABLED=0 go build -o "$bin" ./cmd/serve) || { echo "FATAL: build at $rev failed"; exit 2; }
done
busy() { ps -eo comm= | grep -E '^(go|.*\.test|serve.*)$' | tr '\n' ' '; }
gate() {
  w=0
  while :; do
    l1=$(awk '{print $1}' /proc/loadavg); b=$(busy)
    if awk -v l="$l1" 'BEGIN{exit !(l <= 1.0)}' && [ -z "$b" ]; then return 0; fi
    [ $((w % 60)) -eq 0 ] && echo "$(ts) waiting: load1=$l1 busy=[$b] (${w}s)"
    [ "$w" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s: stopping (the cells so far stand; re-run the rest)"; exit 75; }
    sleep 10; w=$((w + 10))
  done
}
cell() { # arm kv clients pair
  arm=$1 kv=$2 n=$3 p=$4
  case $arm in A) bin=$B/serve-cpu-$OLD;; B) bin=$B/serve-cpu-$NEW;; esac
  gate
  echo "$(ts) cell $arm kv=$kv clients=$n pair=$p load $(cut -d' ' -f1 /proc/loadavg)"
  BENCH_W7_MODEL=$M7 GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_w7_plain.py "$OUT/w7.json" --clients "$n" --engines goinfer \
    --backend cpu --key "${arm}_${kv}_${n}_${p}" --fixed-nonce --max-tokens 32 --serve-args "-kv $kv" --server-log "$OUT/w7-servers.log" \
    || { echo "$(ts) cell $arm $kv $n $p FAILED"; exit 1; }
}
cd "$SRC" || exit 2
{ echo "rev old $OLD new $NEW; tree $(git rev-parse --short HEAD); kernel $(uname -r); governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null); $(nproc) threads"
  sha256sum "$B/serve-cpu-$OLD" "$B/serve-cpu-$NEW"; echo "started $(date '+%F %T %Z')"; } | tee "$OUT/provenance.txt"
for kv in f16 i8; do
  for p in $(seq 1 10); do
    if [ $((p % 2)) -eq 1 ]; then cell A $kv 4 $p; cell B $kv 4 $p; else cell B $kv 4 $p; cell A $kv 4 $p; fi
  done
  for p in 1 2 3; do
    if [ $((p % 2)) -eq 1 ]; then cell A $kv 2 $p; cell B $kv 2 $p; else cell B $kv 2 $p; cell A $kv 2 $p; fi
  done
done
echo "finished $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
python3 "$HERE/grade-p4a.py" "$OUT/w7.json" | tee "$OUT/grade.txt"

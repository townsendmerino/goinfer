#!/bin/bash
# Night job: gate W7-int8 (docs/tasks/task-metal-int8-2026-10.md, "Gate W7-int8"), pre-registered there on 2026-10-04
# before any graded run: MC3's W7 harness on the 1.5B coder at int8int8, one serve binary pinned at b1e6fa4b for both
# arms, old at -max-concurrent 1 (one generation at a time), new at -max-concurrent 4 (the batched int8 step), both
# -kv-sessions 4 -exact-prefill, greedy, --fixed-nonce, a fresh server per cell, 4 then 1 then 2 clients, old/new x 3
# pairs in the order old new new old old new. Graded by MC3's gates.py.
#
# Binary: (cd ~/tmcode/goinfer/metal && CGO_ENABLED=0 go build -o ~/goinfer-bench/metal-int8-2026-10/serve-metal-b1e6fa4b ./cmd/serve)
# Queued with:
#   python3 scripts/night.py add metal-int8-w7 --est 25 --by "Claude (Mac session), Metal int8 W7 served confirmation" \
#     --doc docs/tasks/task-metal-int8-2026-10.md -- bash docs/measurements/metal-int8-2026-10/run-w7-int8.sh
# Logs: ~/goinfer-logs/metal-int8-2026-10/w7/.
set -u
REPO=$HOME/tmcode/goinfer
B=$HOME/goinfer-bench/metal-int8-2026-10
LOG=$HOME/goinfer-logs/metal-int8-2026-10/w7
OLD=$B/serve-metal-b1e6fa4b
NEW=$B/serve-metal-b1e6fa4b
OUT=$LOG/w7-int8-1.5b.json
SRV=$LOG/w7-int8-servers.log
export BENCH_W7_MODEL=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
export BENCH_MIN_FREE_MB=100  # vm_stat "Pages free" only: model mmaps leave it low with most reclaimable
mkdir -p "$LOG"
cd "$REPO" || exit 1
for f in "$OLD" "$NEW"; do [ -x "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f "$BENCH_W7_MODEL" ] || { echo "missing $BENCH_W7_MODEL"; exit 1; }
case "$BENCH_W7_MODEL" in /Volumes/*|/srv/models/*) echo "model on the archive, not the bench set"; exit 1;; esac
ts() { date '+%H:%M:%S'; }
{
  echo "old:      $OLD (sha256 $(shasum -a 256 "$OLD" | cut -c1-16))"
  echo "new:      $NEW (sha256 $(shasum -a 256 "$NEW" | cut -c1-16))"
  echo "model:    $BENCH_W7_MODEL"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(sysctl -n vm.loadavg)"
} | tee "$LOG/provenance.txt"
cell() {
  k=$1; n=$2
  case $k in old*) bin=$OLD; mc=1;; new*) bin=$NEW; mc=4;; esac
  echo "$(ts) cell $k clients=$n bin=$(basename "$bin")" | tee -a "$LOG/provenance.txt"
  GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_w7_plain.py "$OUT" --clients "$n" --engines goinfer --backend metal \
    --key "$k" --fixed-nonce --serve-args "-quant int8int8 -max-concurrent $mc -kv-sessions 4 -exact-prefill" --server-log "$SRV" || { echo "cell $k failed"; exit 1; }
}
for k in old4_1 new4_1 new4_2 old4_2 old4_3 new4_3; do cell $k 4; done
for k in old1_1 new1_1 new1_2 old1_2 old1_3 new1_3; do cell $k 1; done
for k in old2_1 new2_1 new2_2 old2_2 old2_3 new2_3; do cell $k 2; done
# Precondition: 9 batched servers (the new cells), and every server on the native int8 path (18 cells).
bat=$(grep -c 'generations at once, each on its own resident KV slot, decode tokens batched' "$SRV"); nat=$(grep -c 'decode path: metal-resident (int8int8)' "$SRV")
echo "servers: $bat batched (want 9), $nat on metal-resident (int8int8) (want 18)" | tee -a "$LOG/provenance.txt"
python3 docs/measurements/concurrency-mc3-2026-09-26/gates.py "$OUT" | tee "$LOG/gates-output.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"

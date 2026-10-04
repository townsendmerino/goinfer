#!/bin/bash
# Night job: E-P07 amendment 2 (docs/tasks/task-metal-audit-2026-10.md, "E-P07 amendment 2"): the identity re-run with both
# arms on the exact lane (-exact-prefill), written before this run. Otherwise E-P07's first job exactly: W7 on Qwen3-0.6B, old (serve-metal at d50dbbca: Qwen3 one generation at a
# time) against new (at aa924eca: Qwen3 in MC3), -max-concurrent 4 -kv-sessions 4, greedy, --fixed-nonce, a fresh server
# per cell, 4 then 1 then 2 clients, old/new x 3 pairs in the order old new new old old new. Graded by MC3's gates.py.
#
# Binaries:
#   new: (cd ~/tmcode/goinfer/metal && CGO_ENABLED=0 go build -o ~/goinfer-bench/metal-audit-2026-10/serve-metal-ep07new-aa924eca ./cmd/serve)
#   old: the same from a worktree at d50dbbca (with go.work copied in)
# Queued with:
#   python3 scripts/night.py add metal-audit-ep07-exact --est 10 --by "Claude (Mac session), Metal audit E-P07 identity re-run, exact lane" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash docs/measurements/metal-audit-2026-10/run-ep07-w7-exact.sh
# Logs: ~/goinfer-logs/metal-audit-2026-10/ep07-exact/.
set -u
REPO=$HOME/tmcode/goinfer
B=$HOME/goinfer-bench/metal-audit-2026-10
LOG=$HOME/goinfer-logs/metal-audit-2026-10/ep07-exact
OLD=$B/serve-metal-ep07old-d50dbbca
NEW=$B/serve-metal-ep07new-aa924eca
OUT=$LOG/w7-qwen3.json
SRV=$LOG/w7-qwen3-servers.log
export BENCH_W7_MODEL=$HOME/models/qwen3-0.6b-bf16
export BENCH_MIN_FREE_MB=100  # vm_stat "Pages free" only: model mmaps leave it low with most reclaimable
mkdir -p "$LOG"
cd "$REPO" || exit 1
for f in "$OLD" "$NEW"; do [ -x "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f "$BENCH_W7_MODEL/config.json" ] || { echo "missing $BENCH_W7_MODEL"; exit 1; }
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
  case $k in old*) bin=$OLD;; new*) bin=$NEW;; esac
  echo "$(ts) cell $k clients=$n bin=$(basename "$bin")" | tee -a "$LOG/provenance.txt"
  GOINFER_SERVE_CPU=$bin python3 -u scripts/bench_w7_plain.py "$OUT" --clients "$n" --engines goinfer --backend metal \
    --key "$k" --fixed-nonce --serve-args "-max-concurrent 4 -kv-sessions 4 -exact-prefill" --server-log "$SRV" || { echo "cell $k failed"; exit 1; }
}
for k in old4_1 new4_1 new4_2 old4_2 old4_3 new4_3; do cell $k 4; done
for k in old1_1 new1_1 new1_2 old1_2 old1_3 new1_3; do cell $k 1; done
for k in old2_1 new2_1 new2_2 old2_2 old2_3 new2_3; do cell $k 2; done
# Precondition: every server announced its arm's concurrency (9 one-at-a-time, 9 batched).
one=$(grep -c 'concurrency: one generation at a time' "$SRV"); bat=$(grep -c 'generations at once, each on its own resident KV slot, decode tokens batched' "$SRV")
echo "servers: $one one-at-a-time, $bat batched (want 9 and 9)" | tee -a "$LOG/provenance.txt"
python3 docs/measurements/concurrency-mc3-2026-09-26/gates.py "$OUT" | tee "$LOG/gates-output.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"

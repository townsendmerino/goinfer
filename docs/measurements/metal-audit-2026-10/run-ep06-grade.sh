#!/bin/bash
# Night job: E-P06's grade (docs/tasks/task-metal-audit-2026-10.md, "E-P06: built, pending its grade"), pre-registered
# there on 2026-10-02 before any graded run. Served, one client, greedy, the spec record's two workloads
# (docs/measurements/metal-spec-step-verify-2026-09-27.md): copy (scripts/bench_spec_copy.py) and W7 chat
# (scripts/bench_w7_plain.py, --fixed-nonce). Arms rotate per round, 5 rounds on the 1.5B and 3 on the others: plain (no spec), specold (--spec ngram,
# the shipped verify cost constant: serve at REV_OLD, E-P06's parent) and specnew (--spec ngram, the model's own curve
# measured at load: serve at REV_NEW). Graded by gates-ep06.py: chat on the 1.5B, spec-new / spec-old.
#
# Pinned binaries, built by day (the tree may move before tonight):
#   (cd <worktree at REV> && cd metal && CGO_ENABLED=0 go build -o ~/goinfer-bench/metal-audit-2026-10/serve-metal-ep06<arm>-<REV> ./cmd/serve)
# Queued with:
#   python3 scripts/night.py add metal-audit-ep06 --est 60 --by "Claude (Mac session), Metal audit E-P06 grade" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-ep06-grade.sh
# The runner holds the timing lock. Logs and JSON: ~/goinfer-logs/metal-audit-2026-10/ep06/.
set -u
REV_NEW=2bbd495c
REV_OLD=e3353def
R=$HOME/tmcode/goinfer-metal-audit
B=$HOME/goinfer-bench/metal-audit-2026-10
V=$HOME/goinfer-logs/metal-audit-2026-10/ep06
NEW=$B/serve-metal-ep06new-$REV_NEW
OLD=$B/serve-metal-ep06old-$REV_OLD
M15=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
M05=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
M7=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf
mkdir -p "$V"
for f in "$NEW" "$OLD"; do [ -x "$f" ] || { echo "missing binary $f"; exit 1; }; done
for f in "$M15" "$M05" "$M7"; do [ -f "$f" ] || { echo "missing model $f"; exit 1; }; done
cd "$R" || exit 1
export BENCH_MIN_FREE_MB=100
ts() { date '+%H:%M:%S'; }
{
  echo "rev new:  $REV_NEW ($NEW, sha256 $(shasum -a 256 "$NEW" | cut -c1-16))"
  echo "rev old:  $REV_OLD ($OLD, sha256 $(shasum -a 256 "$OLD" | cut -c1-16))"
  echo "started:  $(date '+%F %T %Z')"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "load:     $(sysctl -n vm.loadavg)"
  echo "therm:    $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
} | tee "$V/provenance.txt"
# The idle gate is bench_peer.py's instant gate (the darwin default since TE1): the CPU under BENCH_MAX_BUSY percent busy
# over 3 s and no foreign timed workload, waiting up to 30 min. The first night's run (2026-10-02) waited for a load
# average <= 1.0, which this Mac never reaches with VS Code open, and failed after 30 min without a cell.
gate() { python3 - "$R/scripts" <<'PY' || { echo "$(ts) NOT IDLE after 1800s — stopping" | tee -a "$V/provenance.txt"; exit 1; }
import sys, time
sys.path.insert(0, sys.argv[1])
import bench_peer as b
t0 = time.time()
while True:
    busy, active = b.instant_idle_sample()
    if busy <= b.BUSY_CAP and not active:
        sys.exit(0)
    if time.time() - t0 > 1800:
        sys.exit(1)
    if int(time.time() - t0) % 60 < 4:
        print(f"waiting for idle: busy {busy:.1f}% (cap {b.BUSY_CAP:.0f}%), foreign {active}", flush=True)
    time.sleep(1)
PY
}
bin() { case $1 in specold) echo "$OLD";; *) echo "$NEW";; esac; }
args() { case $1 in plain) echo "";; *) echo "--serve-args=-spec=ngram";; esac; }
copy() { gate; echo "$(ts) copy $2 ${3}_$4"
  GOINFER_SERVE_CPU=$(bin $3) BENCH_W7_MODEL=$1 python3 -u scripts/bench_spec_copy.py "$V/copy-$2.json" --key ${3}_$4 --clients 1 $(args $3) --server-log "$V/servers.log"; }
chat() { gate; echo "$(ts) chat $2 ${3}_$4"
  GOINFER_SERVE_CPU=$(bin $3) BENCH_W7_MODEL=$1 python3 -u scripts/bench_w7_plain.py "$V/chat-$2.json" --clients 1 --engines goinfer --backend metal --key ${3}_$4 --fixed-nonce $(args $3) --server-log "$V/servers.log"; }
ORDER=("plain specold specnew" "specnew plain specold" "specold specnew plain")
for tag in 15b 05b 7b; do
  case $tag in 15b) M=$M15; N=5;; 05b) M=$M05; N=3;; 7b) M=$M7; N=3;; esac # 5 rounds on the graded 1.5B
  for r in $(seq 1 $N); do for arm in ${ORDER[$(((r-1)%3))]}; do copy "$M" $tag $arm $r; done; done
  for r in $(seq 1 $N); do for arm in ${ORDER[$(((r-1)%3))]}; do chat "$M" $tag $arm $r; done; done
done
python3 "$R/docs/measurements/metal-audit-2026-10/gates-ep06.py" "$V" | tee "$V/results.txt"
echo "finished: $(date '+%F %T %Z')" | tee -a "$V/provenance.txt"

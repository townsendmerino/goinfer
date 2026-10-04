#!/usr/bin/env bash
# Night job: the serve chain's served grade (docs/tasks/task-metal-audit-2026-10.md, "The serve chain: served grade"),
# registered 2026-10-04 before it runs. TE5(a): two arms, no peer. goinfer = main 3b9ef839 (a lone MC3 generation runs
# the greedy or sampled chain with the resident held, 0277f4ed); goinfer_old = the same rev with the hold never taken
# (servechain-old-arm.patch, the gate's "never hold" mutation), so the arms differ in that one branch. bench_peer.py on
# Metal at serve's defaults (2 KV slots, so every generation takes an MC3 place), depth 128, greedy (phase A) and
# temp1.0_notrunc (phase C), the 0.5B, 1.5B and 7B at int4, 3 runs per cell. Four passes, the engine order alternating
# (new first, old first, new first, old first). servechain_cells.py tabulates them and checks the precondition from
# each cell's serve log.
#
# Pinned (the tree may move before tonight):
#   serve binaries ~/goinfer-bench/metal-audit-2026-10/serve-metal-servechain{,-nohold}-3b9ef839
#   scripts worktree ~/goinfer-bench/metal-audit-2026-10/wt-servechain-3b9ef839, checked out at fca400fe (bench_peer.py, prompts.json,
#     servechain_cells.py; fca400fe is 3b9ef839 plus docs, and its one local diff is the old arm patch, in decoder/model.go)
# Queued with:
#   python3 scripts/night.py add metal-audit-servechain --est 40 --by "Claude (Mac session), serve-chain grade" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash docs/measurements/metal-audit-2026-10/run-servechain-grade.sh
# bench_peer.py holds the timing lock (inherited from the night runner) and uses the darwin instant idle gate.
set -uo pipefail
REV=3b9ef839
BASE=$HOME/goinfer-bench/metal-audit-2026-10
WT=$BASE/wt-servechain-$REV
LOG=$HOME/goinfer-logs/metal-audit-2026-10/servechain
export GOINFER_SERVE_METAL=$BASE/serve-metal-servechain-$REV GOINFER_SERVE_METAL_OLD=$BASE/serve-metal-servechain-nohold-$REV
export BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_SAMPLED_BACKEND=metal BENCH_DEPTHS=none
export BENCH_CONFIGS=temp1.0_notrunc BENCH_MODELS=0.5B,1.5B,7B BENCH_RUNS=3
mkdir -p "$LOG"
for f in "$GOINFER_SERVE_METAL" "$GOINFER_SERVE_METAL_OLD"; do [ -x "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f "$WT/scripts/bench_peer.py" ] || { echo "missing scripts worktree $WT"; exit 1; }
[ -f "$WT/docs/measurements/metal-audit-2026-10/servechain_cells.py" ] || { echo "missing servechain_cells.py in $WT"; exit 1; }
cd "$WT"
echo "started: $(date '+%F %T %Z'); new $REV (sha256 $(shasum -a 256 "$GOINFER_SERVE_METAL" | cut -c1-16)), old $REV never-hold (sha256 $(shasum -a 256 "$GOINFER_SERVE_METAL_OLD" | cut -c1-16))" | tee "$LOG/provenance.txt"
rc=0
for i in 1 2 3 4; do
  if [ $((i % 2)) -eq 1 ]; then export BENCH_ENGINES=goinfer,goinfer_old; else export BENCH_ENGINES=goinfer_old,goinfer; fi
  mkdir -p "$LOG/pass$i/serve-logs"
  echo "pass $i ($BENCH_ENGINES): $(date '+%T')" | tee -a "$LOG/provenance.txt"
  BENCH_SERVE_LOG_DIR=$LOG/pass$i/serve-logs python3 -u scripts/bench_peer.py "$LOG/pass$i/peer.json" > "$LOG/pass$i/peer.log" 2>&1
  r=$?
  echo "pass $i bench_peer exit $r: $(date '+%T')" | tee -a "$LOG/provenance.txt"
  [ $r -eq 0 ] || rc=$r
done
python3 docs/measurements/metal-audit-2026-10/servechain_cells.py "$LOG" | tee "$LOG/cells.txt"
t=${PIPESTATUS[0]}
[ $t -eq 0 ] || rc=$t
echo "finished: $(date '+%F %T %Z'), exit $rc" | tee -a "$LOG/provenance.txt"
exit $rc

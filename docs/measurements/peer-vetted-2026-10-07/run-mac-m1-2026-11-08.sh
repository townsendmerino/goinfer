#!/bin/bash
# Cell M1 on main after #775 (docs/measurements/peer-vetted-2026-10-07-macbook.md, "M1 on main after #775", re-registered 2026-11-08
# before it runs): run-mac-m1-2026-11-07.sh with three differences, REV (main at 76d78d19, after #775's bbfce11d), the serve's
# registered sha256, and the records' directory (results-mac-m1-2026-11-08). The cell, its rules and grade.py are unchanged. By day:
# the caller holds the timing lock for the whole job and keeps the Mac awake. A refused load, swap growth across the fit check, or a
# battery start exits 75, and the run stops there for the coordinator; nothing retries it or forces the load.
set -u
REV=76d78d19
SERVE_SHA=9a2e74bd1ce5fb28 # the serve built by day, as the record registers it
B=$HOME/goinfer-bench/peer-vetted
TREE=$B/tree-$REV
OUT=$B/results-mac-m1-2026-11-08
PHI3=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
PV=$TREE/docs/measurements/peer-vetted-2026-10-07
export GOINFER_SERVE_METAL=$B/serve-metal-$REV OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
for f in "$PHI3" "$GOINFER_SERVE_METAL" "$OLLAMA_BIN" "$TREE/scripts/bench_peer.py" "$PV/grade.py" "$PV/footprint.py"; do
  [ -e "$f" ] || { echo "VOID: $f is missing"; exit 2; }
done
sha=$(shasum -a 256 "$GOINFER_SERVE_METAL" | cut -c1-16)
[ "$sha" = "$SERVE_SHA" ] || { echo "VOID: the serve's sha256 is $sha, not the registered $SERVE_SHA"; exit 2; }
mkdir -p "$OUT"
{ echo "rev $REV (serve sha256 $sha)"; echo "ollama $($OLLAMA_BIN --version 2>&1 | tail -1)"
  echo "phi3 $(shasum -a 256 "$PHI3" | cut -c1-16)"; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1
  sysctl -n vm.swapusage; df -h ~ | tail -1; uptime; } | tee "$OUT/provenance.txt"
pmset -g batt | head -1 | grep -q 'AC Power' || { echo "on battery: refused"; exit 75; }
# swap_used_mb is system swap in use, in MB, from vm.swapusage ("total = … used = 977.50M free = …").
swap_used_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'; }
# The fit check is the job's first step (owner, 2026-10-16): the pinned server with the cell's flags, read until it serves or refuses.
# The cell runs only on a resident q4k load; a refusal, a weight-streaming fallback or any other decode path ends the job VOID-refused,
# with the figures in $OUT/fitcheck.log.
FIT=$OUT/fitcheck.log
swap0=$(swap_used_mb)
{ echo "=== $(date '+%F %T %Z') / $(date -u +%T) UTC fit check"; sysctl -n vm.swapusage; uptime; } > "$FIT"
"$GOINFER_SERVE_METAL" -model "bench=$PHI3" -backend metal -addr 127.0.0.1:18931 -quant q4k -ctx 2048 >>"$FIT" 2>&1 &
fpid=$!
fit=TIMEOUT
for _ in $(seq 1 600); do # 5 minutes: a cold read of the 2.4 GB file
  if grep -qE 'does not fit resident RAM|NOT loaded' "$FIT"; then fit=REFUSED; break; fi
  if grep -q 'goinfer serving on' "$FIT"; then
    if grep -qE 'decode path: metal-resident \(q4k\)' "$FIT"; then fit=RESIDENT; else fit=NOT-RESIDENT; fi
    { echo "=== $(date '+%T') serving: $(python3 "$PV/footprint.py" $fpid); swap $(sysctl -n vm.swapusage)"; } | tee -a "$FIT"
    break
  fi
  kill -0 $fpid 2>/dev/null || { fit=EXITED; break; }
  sleep 0.5
done
# Stop the check's server; one that ignores SIGTERM for 30 s is killed, so a wedged server cannot hold the queue.
kill $fpid 2>/dev/null
for _ in $(seq 1 60); do kill -0 $fpid 2>/dev/null || break; sleep 0.5; done
kill -0 $fpid 2>/dev/null && { echo "=== the fit check's server ignored SIGTERM for 30 s: SIGKILL" | tee -a "$FIT"; kill -9 $fpid 2>/dev/null; }
wait $fpid 2>/dev/null
echo "=== fit check: $fit" | tee -a "$FIT"
if [ "$fit" != RESIDENT ]; then
  grep -E 'decode path|needs ~|budget|does not fit|weight-streaming retry|refus' "$FIT"
  echo "VOID-refused: the fit check read $fit, not a resident q4k load, so the cell did not run (record: $FIT)"
  exit 75
fi
sleep 5 # the check's memory is released before the cell's own loads
# The registered swap rules, applied before the cell starts: any growth across the fit check is the void's 0 MB bar, which the cell
# would hit too, so the cell is refused for the queue to retry rather than started and voided; growth past 1 GB is the kill's.
swap1=$(swap_used_mb)
grew=$(python3 -c "print(round(float('$swap1') - float('$swap0'), 2))")
echo "=== swap across the fit check: $swap0 MB to $swap1 MB ($grew MB)" | tee -a "$FIT"
if python3 -c "import sys; sys.exit(0 if float('$grew') > 0 else 1)"; then
  echo "VOID-refused: swap grew $grew MB across the fit check (the void's bar is 0 MB; the kill's 1,024 MB), so the cell did not start"
  exit 75
fi
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama BENCH_DEPTHS=none BENCH_BACKENDS=metal BENCH_CTX=2048
export BENCH_QUANT_OVERRIDE="M35=int4mix,phi3-mini=q4k,G20=default,G26Q=default"
export BENCH_SWAP_VOID_MB=0 BENCH_SWAP_KILL_MB=1024 BENCH_SERVE_LOG_DIR=$OUT/serve-logs
cd "$TREE" || exit 1
echo "=== $(date -u +%FT%TZ) START Phi-3 q4k on Metal"
BENCH_MODELS=phi3-mini python3 -u scripts/bench_peer.py "$OUT/metal-phi3-q4k.json"; rc=$?
echo "=== $(date -u +%FT%TZ) END rc=$rc; swap $(sysctl -n vm.swapusage)"
if [ "$rc" != 0 ]; then
  echo "bench_peer.py exited $rc: not graded"
  exit $rc
fi
# grade.py's stdout (a "## file" header, then a JSON line a row) is kept whole in grade.txt and its JSON lines alone in grade.jsonl;
# its stderr goes to its own log, so a warning cannot corrupt either; its failure fails the job.
python3 "$PV/grade.py" "$OUT" > "$OUT/grade.txt" 2> "$OUT/grade.stderr"; grc=$?
grep '^{' "$OUT/grade.txt" > "$OUT/grade.jsonl"
cat "$OUT/grade.txt"
if [ "$grc" != 0 ]; then
  echo "grade.py exited $grc (stderr: $OUT/grade.stderr)"
  exit 1
fi
exit 0

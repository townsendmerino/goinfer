#!/bin/bash
# Cell M1 on main after #722 (docs/measurements/peer-vetted-2026-10-07-macbook.md, "M1 on main after #722", registered 2026-11-07
# before it runs): the 2026-10-18 cell otherwise unchanged. Phi-3 mini 4k at --quant q4k on Metal against Ollama (p3m-local), depth
# 128, greedy, 3 runs, context 2048, under the record's rules (the swap void, the +1 GB kill, the decode-path gate, GOINFER-ALONE,
# grade.py). Differences from run-mac-m1-fix.sh: REV; the serve and harness come from an archive of REV built by day (this job
# builds nothing and runs no git command); the records go to results-mac-m1-2026-11-07. Night queue only (the queue holds the timing
# lock). The fit guard stays on; a refused load is VOID.
set -u
REV=1ecbb5d6
B=$HOME/goinfer-bench/peer-vetted
TREE=$B/tree-$REV
OUT=$B/results-mac-m1-2026-11-07
PHI3=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
export GOINFER_SERVE_METAL=$B/serve-metal-$REV OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
for f in "$PHI3" "$GOINFER_SERVE_METAL" "$TREE/scripts/bench_peer.py" "$TREE/docs/measurements/peer-vetted-2026-10-07/grade.py"; do
  [ -e "$f" ] || { echo "VOID: $f is missing"; exit 2; }
done
mkdir -p "$OUT"
{ echo "rev $REV (serve sha256 $(shasum -a 256 "$GOINFER_SERVE_METAL" | cut -c1-16))"; echo "ollama $($OLLAMA_BIN --version 2>&1 | tail -1)"
  echo "phi3 $(shasum -a 256 "$PHI3" | cut -c1-16)"; sw_vers | tr '\n' ' '; echo; pmset -g batt | head -1
  sysctl -n vm.swapusage; df -h ~ | tail -1; uptime; } | tee "$OUT/provenance.txt"
pmset -g batt | head -1 | grep -q 'AC Power' || { echo "on battery: refused"; exit 75; }
# The fit check is the job's first step (owner, 2026-10-16): the pinned server with the cell's flags, read until it serves or refuses.
# The cell runs only on a resident q4k load; a refusal, a weight-streaming fallback or any other decode path ends the job VOID-refused,
# with the figures in $OUT/fitcheck.log.
FIT=$OUT/fitcheck.log
{ echo "=== $(date '+%F %T %Z') / $(date -u +%T) UTC fit check"; sysctl -n vm.swapusage; uptime; } > "$FIT"
"$GOINFER_SERVE_METAL" -model "bench=$PHI3" -backend metal -addr 127.0.0.1:18931 -quant q4k -ctx 2048 >>"$FIT" 2>&1 &
fpid=$!
fit=TIMEOUT
for _ in $(seq 1 600); do # 5 minutes: a cold read of the 2.4 GB file
  if grep -qE 'does not fit resident RAM|NOT loaded' "$FIT"; then fit=REFUSED; break; fi
  if grep -q 'goinfer serving on' "$FIT"; then
    if grep -qE 'decode path: metal-resident \(q4k\)' "$FIT"; then fit=RESIDENT; else fit=NOT-RESIDENT; fi
    { echo "=== $(date '+%T') serving: $(python3 "$TREE/docs/measurements/peer-vetted-2026-10-07/footprint.py" $fpid); swap $(sysctl -n vm.swapusage)"; } | tee -a "$FIT"
    break
  fi
  kill -0 $fpid 2>/dev/null || { fit=EXITED; break; }
  sleep 0.5
done
kill $fpid 2>/dev/null; wait $fpid 2>/dev/null
echo "=== fit check: $fit" | tee -a "$FIT"
if [ "$fit" != RESIDENT ]; then
  grep -E 'decode path|needs ~|budget|does not fit|weight-streaming retry|refus' "$FIT"
  echo "VOID-refused: the fit check read $fit, not a resident q4k load, so the cell did not run (record: $FIT)"
  exit 75
fi
sleep 5 # the check's memory is released before the cell's own loads
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama BENCH_DEPTHS=none BENCH_BACKENDS=metal BENCH_CTX=2048
export BENCH_QUANT_OVERRIDE="M35=int4mix,phi3-mini=q4k,G20=default,G26Q=default"
export BENCH_SWAP_VOID_MB=0 BENCH_SWAP_KILL_MB=1024 BENCH_SERVE_LOG_DIR=$OUT/serve-logs
cd "$TREE" || exit 1
echo "=== $(date -u +%FT%TZ) START Phi-3 q4k on Metal"
BENCH_MODELS=phi3-mini python3 -u scripts/bench_peer.py "$OUT/metal-phi3-q4k.json"; rc=$?
echo "=== $(date -u +%FT%TZ) END rc=$rc; swap $(sysctl -n vm.swapusage)"
python3 "$TREE/docs/measurements/peer-vetted-2026-10-07/grade.py" "$OUT" 2>&1 | tee "$OUT/grade.jsonl"
exit $rc

#!/usr/bin/env bash
# G-31a (docs/tasks/task-multimodal-support-2026-10.md, "Gemma 4 31B on nobara ... G-31a", registered before any code): Gemma 4 31B on the CPU.
#   1. the sibling's scale: decoder.test TestQuantConsistency_real on ~/models/gemma-4-E4B-it (plain bf16): int4 against its own int8int8        -> qc-e4b.json
#   2. the 31B's int4 bundle (prequant -quant int4 -embed-int4 -target cpu-amd64), skipped when it exists
#   3. G-31a1, the served smoke: serve --backend cpu on that bundle (+ --vision the checkpoint), three text requests and one image request; the automatic bars  -> smoke/
#   4. the 31B's int8int8 bundle (the int8int8 arm reads it, because quantising 59 GB of bf16 inside the test is not known to fit)
#   5. G-31a2 on the 31B: the same test, arm 1 the int8int8 bundle, arm 2 the int4 bundle                                                           -> qc-31b.json
#   6. scripts/g31a_grade.py: the registered rule. The int8int8 bundle (33 GB) is deleted after step 5 wrote its json; the int4 bundle stays (G-31b needs it).
# Pinned binaries in $BIN (built from main at the rev in $BIN/rev): prequant, serve-cpu, decoder.test. Checkpoints from ~/models (local NVMe), never the archive.
# Estimate: sibling ~15 min, int4 bundle 10-20, smoke ~10, int8 bundle 10-20, 31B A2 ~50-60: 100-130 minutes. Queue:
#   python3 scripts/night.py add g31a --est 150 --by "nobara session, 31B" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/run-g31a-night.sh
# G31A_DRY=1 checks the preconditions and prints the plan without running a step (the plumbing control, also against an empty dir). A step that fails is recorded and the later ones still run where they can.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${BIN:-$HOME/goinfer-bench/s31}
OUT=${1:-$HOME/goinfer-logs/g31a-$(date +%F)}
DIR31=$HOME/models/gemma-4-31B-it
DIRE4=$HOME/models/gemma-4-E4B-it
GIW4=$HOME/models/gemma-4-31B-it.int4.cpu-amd64.giw
GIW8=$HOME/models/gemma-4-31B-it.int8int8.cpu-amd64.giw
PORT=${G31A_PORT:-18470}
fatal() { echo "FATAL: $*" >&2; exit 2; }
for f in prequant serve-cpu decoder.test; do [ -x "$BIN/$f" ] || fatal "$BIN/$f is missing"; done
for d in "$DIR31" "$DIRE4"; do [ -d "$d" ] || fatal "$d is missing"; done
for f in "$DIR31" "$DIRE4" "$OUT"; do case "$f" in /srv/models*|/Volumes/*) fatal "$f is the archive (CLAUDE.md)";; esac; done
free_gb=$(df -BG --output=avail "$HOME/models" | tail -1 | tr -dc '0-9')
need=55; [ -e "$GIW4" ] && need=36
[ "$free_gb" -ge "$need" ] || fatal "only ${free_gb} GB free under ~/models, need $need (the two bundles are ~18 + ~33 GB)"
mkdir -p "$OUT/smoke"
SUM="$OUT/summary.txt"; : > "$SUM"
{ echo "binaries: $BIN (main $(cat "$BIN/rev" 2>/dev/null))"; echo "started: $(date '+%F %T %Z')"; echo "free under ~/models: ${free_gb} GB; load: $(cat /proc/loadavg)"; free -g | sed -n 2p; } | tee "$OUT/provenance.txt"
if [ -n "${G31A_DRY:-}" ]; then echo "DRY: preconditions hold; plan = sibling, int4 bundle, smoke, int8 bundle, 31B A2, grade"; exit 0; fi

step() { # NAME CMD... : output to $OUT/NAME.log, a heartbeat every 60 s, one summary line
  local name=$1; shift
  local t0=$SECONDS rc
  echo "[$(date +%T)] START $name"
  ( while sleep 60; do echo "[$(date +%T)] ... $name running $((SECONDS - t0))s; last: $(tail -n1 "$OUT/$name.log" | cut -c1-120)"; done ) &
  local tick=$!
  "$@" > "$OUT/$name.log" 2>&1
  rc=$?
  kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
  printf '%-22s rc=%-3s %5ss\n' "$name" "$rc" "$((SECONDS - t0))" | tee -a "$SUM"
  return $rc
}
qc() { # LABEL OUTJSON ENV... : the consistency test binary
  local label=$1 json=$2; shift 2
  ( cd "$SRC/decoder" && env GOINFER_HEAVY_TESTS=1 GOINFER_QC_LABEL="$label" GOINFER_QC_OUT="$json" "$@" "$BIN/decoder.test" -test.run '^TestQuantConsistency_real$' -test.v -test.timeout 120m )
}


# 1. the sibling's scale
step qc-e4b qc e4b-bf16 "$OUT/qc-e4b.json" GOINFER_QC_DIR="$DIRE4" || echo "!! the sibling run failed: G-31a2 cannot be graded" | tee -a "$SUM"
grep -E "DONE|--- (PASS|FAIL)" "$OUT/qc-e4b.log" | tail -3

# 2. the int4 bundle
if [ -e "$GIW4" ]; then echo "exists: $GIW4" | tee -a "$SUM"
else step prequant-int4 "$BIN/prequant" -quant int4 -embed-int4 -target cpu-amd64 -o "$GIW4" "$DIR31" || { echo "!! the int4 bundle failed; removing a partial file" | tee -a "$SUM"; rm -f "$GIW4"; }; fi
ls -la "$GIW4" 2>/dev/null | tee -a "$SUM"

# 3. G-31a1, the served smoke
if [ -e "$GIW4" ]; then
  "$BIN/serve-cpu" --model "$GIW4" --vision "$DIR31" --backend cpu --kv-sessions 1 --addr 127.0.0.1:$PORT > "$OUT/smoke/serve.log" 2>&1 < /dev/null &
  spid=$!
  up=0
  for _ in $(seq 1 1800); do
    curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && { up=1; break; }
    kill -0 $spid 2>/dev/null || break
    sleep 1
  done
  if [ "$up" = 1 ]; then
    grep -E "decode path|prefill path|vision" "$OUT/smoke/serve.log" | cut -c1-200 | tee -a "$SUM"
    step smoke python3 "$SRC/scripts/g31a_smoke.py" $PORT "$OUT/smoke" "$SRC" || echo "!! G-31a1 automatic bars NOT held (or a request failed)" | tee -a "$SUM"
    cat "$OUT/smoke.log" | cut -c1-400
  else echo "!! serve did not come up (see $OUT/smoke/serve.log)" | tee -a "$SUM"; fi
  kill $spid 2>/dev/null; wait $spid 2>/dev/null; sleep 20
fi

# 4. the int8int8 bundle
if [ -e "$GIW8" ]; then echo "exists: $GIW8" | tee -a "$SUM"
else step prequant-int8 "$BIN/prequant" -quant int8int8 -target cpu-amd64 -o "$GIW8" "$DIR31" || { echo "!! the int8int8 bundle failed; removing a partial file" | tee -a "$SUM"; rm -f "$GIW8"; }; fi

# 5. G-31a2 on the 31B
if [ -e "$GIW4" ] && [ -e "$GIW8" ]; then
  step qc-31b qc gemma4-31b "$OUT/qc-31b.json" GOINFER_QC_DIR="$DIR31" GOINFER_QC_INT8="$GIW8" GOINFER_QC_INT4="$GIW4" || echo "!! the 31B consistency run failed" | tee -a "$SUM"
  grep -E "DONE|--- (PASS|FAIL)" "$OUT/qc-31b.log" | tail -3
fi

# 6. grade
if [ -e "$OUT/qc-e4b.json" ] && [ -e "$OUT/qc-31b.json" ]; then
  python3 "$SRC/scripts/g31a_grade.py" "$OUT/qc-e4b.json" "$OUT/qc-31b.json" | tee -a "$SUM"
  rm -f "$GIW8" && echo "deleted the int8int8 bundle (the int4 bundle stays)" | tee -a "$SUM"
else echo "!! not graded: a consistency json is missing; the int8int8 bundle is kept for the re-run" | tee -a "$SUM"; fi
echo "finished: $(date '+%F %T %Z')" | tee -a "$OUT/provenance.txt"
exit 0

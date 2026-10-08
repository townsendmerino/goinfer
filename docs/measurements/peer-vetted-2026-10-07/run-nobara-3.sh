#!/bin/bash
# The THIRD run (option (b), re-registered 2026-10-08: swap-ins > 100 MB or growth > 256 MB voids an arm) of nobara's two cells that produced no graded cell on night 1 (registered in
# docs/measurements/peer-vetted-2026-10-07-nobara-pc.md, "The re-run", before it ran): 1c, the vetted 26B on the CPU,
# and 2, gpt-oss on CUDA at ctx 2048. Unchanged from run-nobara.sh but for the harness: it runs from this script's own
# checkout, which has the calibrated G26Q:128 prompt (night 1 crashed on its absence). Serve binaries: the same 65b2c22a builds.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)   # the job's checkout: its scripts/bench_peer.py and scripts/prompts.json
export PATH=$PATH:/usr/local/go/bin
REV=65b2c22a
W=$HOME/goinfer-bench/peer-vetted-2026-10-07
OUT=$W/results-3
ts() { date -u +%FT%TZ; }
mkdir -p "$OUT"
for b in "$W/serve-cuda-$REV" "$W/serve-cpu-$REV"; do [ -x "$b" ] || { echo "FATAL: $b is missing"; exit 2; }; done
python3 -c "import json,sys; json.load(open('$ROOT/scripts/prompts.json'))['G26Q:128']" || { echo "FATAL: no G26Q:128 prompt in $ROOT"; exit 2; }
{ echo "serve rev $REV; harness $(git -C "$ROOT" rev-parse --short HEAD)"; echo "host $(hostname) $(uname -sr)"
  nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
  echo "ollama $($HOME/ollama-0325/bin/ollama --version 2>&1 | tail -1)"; echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
  free -m | sed -n 3p; sha256sum "$W/serve-cuda-$REV" "$W/serve-cpu-$REV"; } | tee "$OUT/provenance.txt"
grep -q "595.91.07" "$OUT/provenance.txt" || { echo "driver is not 595.91.07: a re-anchor, not a carry-forward. Stopping."; exit 1; }
export GOINFER_SERVE_CUDA=$W/serve-cuda-$REV GOINFER_SERVE_CPU=$W/serve-cpu-$REV
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
unset LD_LIBRARY_PATH
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama BENCH_DEPTHS=none
export BENCH_QUANT_OVERRIDE="M35=int4mix,phi3-mini=default,G20=default,G26Q=default"
export BENCH_SWAP_VOID_MB=256 BENCH_SWAPIN_VOID_MB=100 BENCH_SERVE_LOG_DIR=$OUT/serve-logs
cd "$ROOT" || exit 1
H=scripts/bench_peer.py
waitidle() { for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && return; echo "=== waiting for idle: loadavg $l"; sleep 20; done; }
step() { waitidle; echo "=== $(ts) START $1"; t0=$(date +%s); }
done_() { echo "=== $(ts) END $1 rc=$2 ($(( $(date +%s) - t0 ))s)"; }
step "1c CPU: G26Q @128"
BENCH_MODELS=G26Q BENCH_BACKENDS=cpu python3 -u $H "$OUT/cpu-g26q.json"; done_ 1c $?
step "2 CUDA: G20 @128, BENCH_CTX=2048"
BENCH_CTX=2048 BENCH_MODELS=G20 BENCH_BACKENDS=cuda python3 -u $H "$OUT/cuda-g20.json"; done_ 2 $?
python3 "$HERE/grade.py" "$OUT" 2>&1 | tee "$OUT/grade.jsonl"
echo "=== $(ts) ALL DONE"

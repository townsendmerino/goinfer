#!/bin/bash
# Vetted-checkpoint peer cells on nobara-pc, at one pinned commit. Pre-registered in
# docs/measurements/peer-vetted-2026-10-07-nobara-pc.md (committed before this ran). Night queue only; the queue holds
# the timing lock. Cells, all depth 128, greedy, 3 runs, goinfer against Ollama v0.32.5, each engine at its defaults:
#   1. CPU: phi3-mini (goinfer --quant default = q4k), gpt-oss-20b (G20), the vetted 26B (G26Q). Ollama CPU-forced (num_gpu 0).
#   2. CUDA: gpt-oss-20b at ctx 2048 (BENCH_CTX, as cell c): goinfer -moe-cache-experts, Ollama its own partial offload.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)   # the grader, resolved before the cd below
export PATH=$PATH:/usr/local/go/bin
REV=65b2c22a
R=$HOME/mycode/goinfer
W=$HOME/goinfer-bench/peer-vetted-2026-10-07
OUT=$W/results
ts() { date -u +%FT%TZ; }
mkdir -p "$OUT"
[ -d "$W/wt-$REV" ] || git -C "$R" worktree add --detach "$W/wt-$REV" "$REV" || exit 1
[ "$(git -C "$W/wt-$REV" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
if [ ! -x "$W/serve-cuda-$REV" ] || [ ! -x "$W/serve-cpu-$REV" ]; then
  printf 'go 1.27.0\n\nuse (\n\t.\n\t./cuda\n\t./gpu\n)\n' > "$W/wt-$REV/go.work"
  (cd "$W/wt-$REV/cuda" && CGO_ENABLED=0 go build -tags cuda -o "$W/serve-cuda-$REV" ./cmd/serve) || exit 1
  (cd "$W/wt-$REV" && CGO_ENABLED=0 go build -o "$W/serve-cpu-$REV" ./cmd/serve) || exit 1
fi
{ echo "rev $REV"; echo "host $(hostname) $(uname -sr)"; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
  echo "ollama $($HOME/ollama-0325/bin/ollama --version 2>&1 | tail -1)"; echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
  free -m | sed -n 3p; sha256sum "$W/serve-cuda-$REV" "$W/serve-cpu-$REV"; } | tee "$OUT/provenance.txt"
grep -q "595.91.07" "$OUT/provenance.txt" || { echo "driver is not 595.91.07: a re-anchor, not a carry-forward. Stopping."; exit 1; }

export GOINFER_SERVE_CUDA=$W/serve-cuda-$REV GOINFER_SERVE_CPU=$W/serve-cpu-$REV
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
unset LD_LIBRARY_PATH
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama BENCH_DEPTHS=none
export BENCH_QUANT_OVERRIDE="M35=int4mix,phi3-mini=default,G20=default,G26Q=default"
export BENCH_SWAP_VOID_MB=0 BENCH_SERVE_LOG_DIR=$OUT/serve-logs
cd "$W/wt-$REV" || exit 1
H=scripts/bench_peer.py
waitidle() { for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && return; echo "=== waiting for idle: loadavg $l"; sleep 20; done; }
step() { waitidle; echo "=== $(ts) START $1"; t0=$(date +%s); }
done_() { echo "=== $(ts) END $1 rc=$2 ($(( $(date +%s) - t0 ))s)"; }
step "1 CPU: phi3-mini, G20, G26Q @128"
BENCH_MODELS=phi3-mini,G20,G26Q BENCH_BACKENDS=cpu python3 -u $H "$OUT/cpu.json"; done_ 1 $?
step "2 CUDA: G20 @128, BENCH_CTX=2048"
BENCH_CTX=2048 BENCH_MODELS=G20 BENCH_BACKENDS=cuda python3 -u $H "$OUT/cuda-g20.json"; done_ 2 $?
python3 "$HERE/grade.py" "$OUT" 2>&1 | tee "$OUT/grade.jsonl"
echo "=== $(ts) ALL DONE"

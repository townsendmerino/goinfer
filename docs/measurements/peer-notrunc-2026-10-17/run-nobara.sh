#!/bin/bash
# The no-truncation sampled cells on nobara-pc, re-measured with temperature, top_p and top_k sent to every arm
# (scripts/bench_peer.py since 0600855e). Pre-registered in docs/measurements/peer-notrunc-2026-10-17.md. Cell f's shape
# from docs/measurements/peer-sweep-2026-09-29/run-nobara.sh: 3 runs x 8 x 64 tokens, depth 128, the same peers.
# PREFLIGHT=1 builds and checks the stack, then stops before any timed run. Run from $W, with this record's grade.py copied
# there: the pinned worktree predates this record.
set -u
export PATH=$PATH:/usr/local/go/bin
REV=63038121
R=$HOME/mycode/goinfer
W=$HOME/goinfer-bench/peer-notrunc-2026-10-17
ts() { date -u +%FT%TZ; }
mkdir -p "$W"
echo "$(ts) provenance"; {
  echo "rev $REV"; echo "host $(hostname) $(uname -sr)"; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
  echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "ollama $($HOME/ollama-0325/bin/ollama --version 2>&1 | tail -1)"
  echo "llama-server $($HOME/mycode/peers/llama.cpp/build/bin/llama-server --version 2>&1 | head -1)"
} | tee "$W/provenance.txt"
grep -q "595.91.07" "$W/provenance.txt" || { echo "driver is not 595.91.07: that is a re-anchor, not a carry-forward. Stopping."; exit 1; }

echo "$(ts) worktree at $REV"
git -C "$R" fetch -q origin || exit 1
[ -d "$W/wt-$REV" ] || git -C "$R" worktree add -q --detach "$W/wt-$REV" "$REV" || exit 1
[ "$(git -C "$W/wt-$REV" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
printf 'go 1.27.0\n\ntoolchain go1.27.2\n\nuse (\n\t.\n\t./cuda\n\t./gpu\n)\n' > "$W/wt-$REV/go.work"
echo "$(ts) build"
(cd "$W/wt-$REV/cuda" && GOWORK=$W/wt-$REV/go.work CGO_ENABLED=0 go build -tags cuda -o "$W/serve-cuda-$REV" ./cmd/serve) || exit 1

export GOINFER_SERVE_CUDA=$W/serve-cuda-$REV GOINFER_SERVE=$W/serve-cuda-$REV
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
unset LD_LIBRARY_PATH   # Ollama's lib dir made llama-server load the wrong libllama (peer sweep 2026-09-25)
export LLAMACPP_BIN=$HOME/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama,llamacpp BENCH_QUANT_OVERRIDE="phi3-mini=default"
export BENCH_SERVE_LOG_DIR=$W/serve-logs
cd "$W/wt-$REV" || exit 1
python3 -c "
import sys; sys.path.insert(0, 'scripts'); import bench_peer as b
c = b.CONFIGS['temp1.0_notrunc']
assert all(k in c[e] for e in ('goinfer', 'ollama', 'llamacpp') for k in ('temperature', 'top_p', 'top_k')), c
print('temp1.0_notrunc sends', {e: c[e] for e in ('goinfer', 'ollama', 'llamacpp')})" || exit 1
[ "${PREFLIGHT:-}" = 1 ] && { echo "$(ts) PREFLIGHT DONE"; exit 0; }

# The build just loaded the box: wait for idle, as the 09-29 script did before each step (the harness refuses above 1.0).
for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && break; echo "=== waiting for idle: loadavg $l"; sleep 20; done
echo "=== $(ts) START notrunc cells"; t0=$(date +%s)
BENCH_MODELS=0.5B,gemma3-1b,phi3-mini BENCH_BACKENDS=cuda BENCH_DEPTHS=none BENCH_CONFIGS=temp1.0_notrunc \
  python3 -u scripts/bench_peer.py "$W/f-sampled.json"
rc=$?
echo "=== $(ts) END rc=$rc ($(( $(date +%s) - t0 ))s)"
python3 "$W/grade.py" "$W" | tee "$W/grade.jsonl"   # this record's grade.py, copied beside this script
exit $rc

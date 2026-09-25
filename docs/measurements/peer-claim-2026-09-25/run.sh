#!/usr/bin/env bash
# Peer claim sweep, cells a–f (docs/measurements/peer-claim-2026-09-25.md). Resumable: each
# invocation's results file is reloaded by bench_peer.py, which skips completed cells.
set -u
cd /home/francis/mycode/goinfer
R=docs/measurements/peer-claim-2026-09-25
B=~/bench-peer-claim
export GOINFER_SERVE_CUDA=$B/serve-cuda-411e7fc4 GOINFER_SERVE_CPU=$B/serve-cpu-411e7fc4 GOINFER_SERVE=$B/serve-cuda-411e7fc4
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
# NO LD_LIBRARY_PATH. Attempt 1 exported Ollama's lib dir here, and llama-server then loaded Ollama's
# bundled libllama (b4d6c7d8f) instead of its own (427291b) and never came up. Ollama finds its own
# libs without it (checked: CUDA0, 217.8 tok/s on q05).
unset LD_LIBRARY_PATH
export LLAMACPP_BIN=$HOME/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama,llamacpp
waitidle() { for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && return; echo "=== waiting for idle: loadavg $l"; sleep 20; done; }
# Wait before EVERY step: step 2 of attempt 3 was refused at loadavg 1.00, the residue of step 1's own servers.
step() { waitidle; echo "=== $(date -u +%FT%TZ) START $1"; t0=$(date +%s); }
done_() { echo "=== $(date -u +%FT%TZ) END $1 rc=$2 ($(( $(date +%s) - t0 ))s)"; if [ "$2" != 0 ]; then echo "=== ABORT: step $1 failed; later steps not run"; exit "$2"; fi; }
# Wait for an idle box before the first step (the harness refuses above loadavg 1.0; attempt 2 was
# refused at 4.56, left over from a check run just before it).
for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && break; echo "=== waiting for idle: loadavg $l"; sleep 20; done

step "1 dense greedy: cuda+cpu @128 (a, e), cuda @2048/3900 (a)"
BENCH_MODELS=0.5B,1.5B,7B BENCH_BACKENDS=cpu,cuda BENCH_DEPTHS=2048,3900 \
  python3 scripts/bench_peer.py $R/a-e-dense.json; done_ 1 $?

step "2 dense greedy @8000, BENCH_CTX=8192 (a)"
BENCH_CTX=8192 BENCH_MODELS=0.5B,1.5B,7B BENCH_BACKENDS=cuda BENCH_DEPTHS=8000 \
  python3 scripts/bench_peer.py $R/a-depth8000.json; done_ 2 $?

step "3 controls gemma3-1b, phi3-mini @128/3900 (b)"
BENCH_MODELS=gemma3-1b,phi3-mini BENCH_BACKENDS=cuda BENCH_DEPTHS=3900 \
  python3 scripts/bench_peer.py $R/b-controls.json; done_ 3 $?

step "4 sampled 0.5B, phi3-mini @128 (f)"
BENCH_MODELS=0.5B,phi3-mini BENCH_BACKENDS=cuda BENCH_DEPTHS=none \
  BENCH_CONFIGS=temp1.0_notrunc,temp0.8_topp0.95 \
  python3 scripts/bench_peer.py $R/f-sampled.json; done_ 4 $?

step "5 26B MoE @128, BENCH_CTX=2048 (c)"
BENCH_CTX=2048 BENCH_MODELS=M26 BENCH_BACKENDS=cuda BENCH_DEPTHS=none \
  python3 scripts/bench_peer.py $R/c-26b.json; done_ 5 $?

step "6 prefill TTFT 1.5B K=512/3900 (d)"
python3 scripts/bench_peer_prefill.py $R/d-prefill.json --models 1.5B --depths 512,3900 --n 6 --backend cuda; done_ 6 $?
echo "=== $(date -u +%FT%TZ) ALL DONE"

#!/bin/bash
# Peer sweep cells a-f on nobara-pc, at one pinned commit. Pre-registered in docs/measurements/peer-sweep-2026-09-29.md.
# Resumable: bench_peer.py reloads each results file and skips completed cells. PREFLIGHT=1 stops before any timed run.
set -u
export PATH=$PATH:/usr/local/go/bin
REV=754f12d3
R=$HOME/mycode/goinfer
W=$HOME/goinfer-bench/peer-sweep-2026-09-29
ts() { date -u +%FT%TZ; }
mkdir -p "$W"
echo "$(ts) provenance"; {
  echo "rev $REV"; echo "host $(hostname) $(uname -sr)"; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
  echo "aikit $(git -C "$HOME/mycode/aikit/aikit" log --oneline -1)"; echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "ollama $($HOME/ollama-0325/bin/ollama --version 2>&1 | tail -1)"
  echo "llama-server $($HOME/mycode/peers/llama.cpp/build/bin/llama-server --version 2>&1 | head -1)"
} | tee "$W/provenance.txt"
grep -q "595.91.07" "$W/provenance.txt" || { echo "driver is not 595.91.07: that is a re-anchor, not a carry-forward. Stopping."; exit 1; }

echo "$(ts) worktree at $REV"
git -C "$R" fetch -q origin || exit 1
[ -d "$W/wt-$REV" ] || git -C "$R" worktree add --detach "$W/wt-$REV" "$REV" || exit 1
[ "$(git -C "$W/wt-$REV" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
cp "$R/go.work" "$W/wt/go.work"; sed -i "s#\.\./aikit/aikit#$HOME/mycode/aikit/aikit#" "$W/wt/go.work"
echo "$(ts) build"
(cd "$W/wt/cuda" && CGO_ENABLED=0 go build -tags cuda -o "$W/serve-cuda-$REV" ./cmd/serve) || exit 1
(cd "$W/wt-$REV" && CGO_ENABLED=0 go build -o "$W/serve-cpu-$REV" ./cmd/serve) || exit 1

export GOINFER_SERVE_CUDA=$W/serve-cuda-$REV GOINFER_SERVE_CPU=$W/serve-cpu-$REV GOINFER_SERVE=$W/serve-cuda-$REV
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
unset LD_LIBRARY_PATH   # 09-25 attempt 1: Ollama's lib dir made llama-server load the wrong libllama
export LLAMACPP_BIN=$HOME/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama,llamacpp BENCH_QUANT_OVERRIDE="M35=int4mix,phi3-mini=default"
export BENCH_SERVE_LOG_DIR=$W/serve-logs
cd "$W/wt-$REV" || exit 1
H=scripts/bench_peer.py

if [ "${PREFLIGHT:-}" = 1 ]; then
  # no timing: one load of each risky goinfer configuration as the decode harness launches it, and the decode path it resolves to
  for cfg in "cuda phi3-mini" "cuda 0.5B"; do set -- $cfg
    case $1/$2 in
      cuda/phi3-mini) M=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf ;;
      *) M=$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf ;;
    esac
    "$GOINFER_SERVE_CUDA" -model "bench=$M" -backend cuda -addr 127.0.0.1:8197 > "$W/preflight-$2.log" 2>&1 &
    pid=$!; for _ in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8197/v1/models >/dev/null 2>&1 && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    echo "preflight $2: $(grep -m1 'decode path' "$W/preflight-$2.log" | cut -c1-160) $(grep -m1 -i 'declined\|not loaded\|refus' "$W/preflight-$2.log" | cut -c1-120)"
  done
  echo "$(ts) PREFLIGHT DONE"; exit 0
fi

waitidle() { for i in $(seq 1 90); do l=$(cut -d' ' -f1 /proc/loadavg); awk -v l="$l" 'BEGIN{exit !(l<0.8)}' && return; echo "=== waiting for idle: loadavg $l"; sleep 20; done; }
step() { waitidle; echo "=== $(ts) START $1"; t0=$(date +%s); }
done_() { echo "=== $(ts) END $1 rc=$2 ($(( $(date +%s) - t0 ))s)"; if [ "$2" != 0 ]; then echo "=== ABORT: step $1 failed; later steps not run"; exit "$2"; fi; }

step "1 dense greedy: cuda+cpu @128 (a, e), cuda @2048/3900 (a)"
BENCH_MODELS=0.5B,1.5B,7B BENCH_BACKENDS=cpu,cuda BENCH_DEPTHS=2048,3900 python3 -u $H "$W/a-e-dense.json"; done_ 1 $?
step "2 dense greedy @8000, BENCH_CTX=8192 (a)"
BENCH_CTX=8192 BENCH_MODELS=0.5B,1.5B,7B BENCH_BACKENDS=cuda BENCH_DEPTHS=8000 python3 -u $H "$W/a-depth8000.json"; done_ 2 $?
step "3 controls gemma3-1b, phi3-mini @128/3900 (b)"
BENCH_MODELS=gemma3-1b,phi3-mini BENCH_BACKENDS=cuda BENCH_DEPTHS=3900 python3 -u $H "$W/b-controls.json"; done_ 3 $?
step "4 sampled 0.5B, phi3-mini @128 (f)"
BENCH_MODELS=0.5B,phi3-mini BENCH_BACKENDS=cuda BENCH_DEPTHS=none BENCH_CONFIGS=temp1.0_notrunc,temp0.8_topp0.95 python3 -u $H "$W/f-sampled.json"; done_ 4 $?
step "5 26B MoE @128, BENCH_CTX=2048 (c)"
BENCH_CTX=2048 BENCH_MODELS=M26 BENCH_BACKENDS=cuda BENCH_DEPTHS=none python3 -u $H "$W/c-26b.json"; done_ 5 $?
step "6 prefill TTFT 1.5B K=512/3900 (d)"
python3 -u scripts/bench_peer_prefill.py "$W/d-prefill.json" --models 1.5B --depths 512,3900 --n 6 --backend cuda; done_ 6 $?
echo "=== $(ts) ALL DONE"

#!/bin/bash
# Peer sweep cells g-i on the MacBook, at one pinned commit. Pre-registered in docs/measurements/peer-sweep-2026-09-29.md.
# By night only. PREFLIGHT=1 builds, and loads each Metal configuration once as the harness launches it; no timing.
set -u
REV=754f12d3
R=$HOME/tmcode/goinfer
B=$HOME/goinfer-bench/peer-sweep-2026-09-29
ts() { date '+%H:%M:%S'; }
mkdir -p "$B"
echo "$(ts) worktree at $REV"
git -C "$R" fetch -q origin || exit 1
[ -d "$B/wt-$REV" ] || git -C "$R" worktree add --detach "$B/wt-$REV" "$REV" || exit 1
[ "$(git -C "$B/wt-$REV" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "worktree is not at $REV"; exit 1; }
cp "$R/go.work" "$B/wt/go.work"; sed -i '' "s#\.\./aikit#$HOME/tmcode/aikit#" "$B/wt/go.work" 2>/dev/null
echo "$(ts) build"
(cd "$B/wt/metal" && CGO_ENABLED=0 go build -o "$B/serve-metal-$REV" ./cmd/serve) || exit 1
(cd "$B/wt-$REV" && CGO_ENABLED=0 go build -o "$B/serve-cpu-$REV" ./cmd/serve) || exit 1
{ echo "rev $REV"; echo "host $(hostname) $(uname -srm)"; sysctl -n machdep.cpu.brand_string; echo "load $(sysctl -n vm.loadavg)"
  echo "ollama $(/opt/homebrew/bin/ollama --version 2>&1 | tail -1)"; echo "llama-server $(/opt/homebrew/bin/llama-server --version 2>&1 | head -1)"
  echo "os $(sw_vers -productVersion)"; } | tee "$B/provenance.txt"

export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models LLAMACPP_BIN=/opt/homebrew/bin/llama-server
export GOINFER_SERVE_METAL=$B/serve-metal-$REV GOINFER_SERVE_CPU=$B/serve-cpu-$REV
export BENCH_RUNS=3 BENCH_ENGINES=goinfer,ollama,llamacpp BENCH_SERVE_LOG_DIR=$B/serve-logs
cd "$B/wt-$REV" || exit 1
H=scripts/bench_peer.py

if [ "${PREFLIGHT:-}" = 1 ]; then
  M=$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
  for flags in "" "-embed-int4=false"; do
    "$GOINFER_SERVE_METAL" -model "bench=$M" -backend metal -quant int4 -addr 127.0.0.1:8197 $flags > "$B/preflight-${flags:-default}.log" 2>&1 &
    pid=$!; for _ in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8197/v1/models >/dev/null 2>&1 && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
    echo "preflight [${flags:-default flags}]: $(grep -m1 'decode path' "$B/preflight-${flags:-default}.log" | cut -c1-150) | $(grep -m1 -i 'declined\|did not resolve' "$B/preflight-${flags:-default}.log" | cut -c1-120)"
  done
  echo "$(ts) PREFLIGHT DONE"; exit 0
fi

# bench_peer.py refuses a busy box and bench_peer_prefill.py reads /proc/loadavg (macOS has none), so wait first.
wait_idle() { local w=0; while :; do l1=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l="$l1" 'BEGIN{exit !(l <= 1.0)}' && { echo "$(ts) idle: load1=$l1"; return 0; }
  [ "$w" -ge 1800 ] && { echo "$(ts) NOT IDLE after 1800s (load1=$l1)"; return 1; }; sleep 10; w=$((w + 10)); done; }

echo "$(ts) == cell g (Metal decode 0.5B/1.5B/7B @ 128/2048/3900), --embed-int4=false start"
wait_idle || exit 1
BENCH_GOINFER_ARGS="-embed-int4=false" BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_DEPTHS=2048,3900 BENCH_MODELS=0.5B,1.5B,7B \
  python3 -u $H "$B/g-metal-decode.json"; echo "$(ts) == cell g exit=$?"
echo "$(ts) == cell i (CPU arm64 decode 0.5B/1.5B @128), shipped defaults start"
wait_idle || exit 1
BENCH_BACKENDS=cpu BENCH_DEPTHS=none BENCH_MODELS=0.5B,1.5B python3 -u $H "$B/i-cpu-decode.json"; echo "$(ts) == cell i exit=$?"
echo "$(ts) == cell h (Metal prefill TTFT 1.5B K=512/3900, 6 prompts), --embed-int4=false start"
wait_idle || exit 1
BENCH_GOINFER_ARGS="-embed-int4=false" python3 -u scripts/bench_peer_prefill.py "$B/h-metal-prefill.json" --models 1.5B --depths 512,3900 --n 6 --backend metal
echo "$(ts) == cell h exit=$?"
echo "$(ts) == DONE"

#!/usr/bin/env bash
# Night job (Mac): the two Mac peer rows that predate the Metal audit (pre-registered in ../peer-refresh-2026-10-06.md,
# committed before this runs). Step 1: phi3-mini Metal decode against Ollama (bench_peer.py; no MLX phi3-mini checkpoint
# here). Step 2: the 1.5B's long-prompt TTFT at K = 512 and 3900 against Ollama and mlx-lm (bench_peer_prefill.py).
# Both run the pinned serve binary built from REV, from a scripts worktree at REV (the tree may move before tonight).
# Queued with:
#   python3 scripts/night.py add peer-refresh-mac --est 45 --by "Claude (Mac session), Mac peer rows" \
#     --doc docs/measurements/peer-refresh-2026-10-06.md -- bash docs/measurements/peer-refresh-2026-10-06/run-mac.sh
# Both harnesses hold the timing lock (inherited from the night runner). bench_peer.py uses the darwin instant idle gate;
# bench_peer_prefill.py has none of its own, so this runs at night with nothing else on the box.
set -uo pipefail
REV=52024b37
BASE=$HOME/goinfer-bench/peer-refresh-2026-10-06
WT=$BASE/wt-$REV
LOG=$HOME/goinfer-logs/peer-refresh-2026-10-06
export GOINFER_SERVE_METAL=$BASE/serve-metal-$REV
export MLX_SERVER_BIN=$HOME/Library/Python/3.12/bin/mlx_lm.server
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
mkdir -p "$LOG/serve-logs-decode" "$LOG/serve-logs-prefill"
for f in "$GOINFER_SERVE_METAL" "$MLX_SERVER_BIN" "$OLLAMA_BIN"; do [ -x "$f" ] || { echo "missing $f"; exit 1; }; done
for f in "$WT/scripts/bench_peer.py" "$WT/scripts/bench_peer_prefill.py"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f "$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf" ] || { echo "missing the phi3-mini GGUF"; exit 1; }
[ -d "$HOME/models/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit" ] || { echo "missing the MLX 1.5B checkpoint"; exit 1; }
cd "$WT"
echo "started: $(date '+%F %T %Z'); goinfer $REV (sha256 $(shasum -a 256 "$GOINFER_SERVE_METAL" | cut -c1-16)); ollama $($OLLAMA_BIN --version 2>&1 | tail -1); mlx-lm $(python3 -c 'import mlx_lm; print(mlx_lm.__version__)')" | tee "$LOG/provenance.txt"

echo "=== step 1: phi3-mini decode, $(date '+%T')" | tee -a "$LOG/provenance.txt"
BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_SAMPLED_BACKEND=metal BENCH_DEPTHS=2048,3900 \
BENCH_CONFIGS=temp1.0_notrunc BENCH_MODELS=phi3-mini BENCH_ENGINES=goinfer,ollama BENCH_RUNS=3 \
BENCH_SERVE_LOG_DIR=$LOG/serve-logs-decode \
  python3 -u scripts/bench_peer.py "$LOG/phi3-decode.json" 2>&1 | tee "$LOG/phi3-decode.log"
rc1=${PIPESTATUS[0]}

echo "=== step 2: 1.5B long-prompt TTFT, $(date '+%T')" | tee -a "$LOG/provenance.txt"
export MLX_MODELS="{\"1.5B\":\"$HOME/models/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit\"}"  # local disk, never the hub
BENCH_SERVE_LOG_DIR=$LOG/serve-logs-prefill \
  python3 -u scripts/bench_peer_prefill.py "$LOG/prefill-long.json" --backend metal --models 1.5B --depths 512,3900 --n 6 2>&1 | tee "$LOG/prefill-long.log"
rc2=${PIPESTATUS[0]}

echo "finished: $(date '+%F %T %Z'), bench_peer exit $rc1, bench_peer_prefill exit $rc2" | tee -a "$LOG/provenance.txt"
[ "$rc1" = 0 ] && [ "$rc2" = 0 ]

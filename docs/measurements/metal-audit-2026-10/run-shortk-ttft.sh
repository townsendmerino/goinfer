#!/usr/bin/env bash
# Night job: T1.11's other half, the short-prompt Metal TTFT rows (docs/tasks/task-metal-audit-2026-10.md, "Short-prompt
# prefill rows: pre-registration", registered 2026-10-05 before it runs). benchmarks.md's "Metal short-prompt prefill
# floor — R3, 2026-09-20" table predates R16's prefill GEMM, R19's prefill attention, A-P01's tile selector and A-P02's
# floor of 16 (A-D01). The same command as that table, on today's build: scripts/bench_peer_prefill.py --backend metal
# --models 1.5B --depths 32,64,128,256 --n 6. Engines interleaved per cell, each server restarted: goinfer_exact
# (--exact-prefill), goinfer (the default, batched from 16 tokens), Ollama 0.32.5 and mlx-lm 0.31.3. No floor override:
# the default now batches every depth here.
#
# Pinned (the tree may move before tonight): serve binary ~/goinfer-bench/metal-audit-2026-10/serve-metal-shortk-REV and
# scripts worktree ~/goinfer-bench/metal-audit-2026-10/wt-shortk-REV (bench_peer_prefill.py, prompts.json).
# Queued with:
#   python3 scripts/night.py add metal-audit-shortk-ttft --est 30 --priority 50 --by "Claude (Mac session), T1.11 short-K" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash docs/measurements/metal-audit-2026-10/run-shortk-ttft.sh
# bench_peer_prefill.py holds the timing lock (inherited from the night runner); it has no idle gate of its own, so it
# runs at night with nothing else on the box. MLX runs as `python3 -m mlx_lm.server` on the local checkpoint.
set -uo pipefail
REV=93d586af
BASE=$HOME/goinfer-bench/metal-audit-2026-10
WT=$BASE/wt-shortk-$REV
LOG=$HOME/goinfer-logs/metal-audit-2026-10/shortk-ttft
export GOINFER_SERVE_METAL=$BASE/serve-metal-shortk-$REV
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export MLX_MODELS="{\"1.5B\":\"$HOME/models/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit\"}"  # local disk, never the hub
export BENCH_SERVE_LOG_DIR=$LOG/serve-logs
mkdir -p "$LOG" "$LOG/serve-logs"
[ -d "$HOME/models/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit" ] || { echo "missing the MLX 1.5B checkpoint"; exit 1; }
python3 -c 'import mlx_lm' || { echo "mlx_lm not importable by python3"; exit 1; }
for f in "$GOINFER_SERVE_METAL" "$OLLAMA_BIN"; do [ -x "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f "$WT/scripts/bench_peer_prefill.py" ] || { echo "missing scripts worktree $WT"; exit 1; }
cd "$WT"
echo "started: $(date '+%F %T %Z'); goinfer $REV (sha256 $(shasum -a 256 "$GOINFER_SERVE_METAL" | cut -c1-16)); ollama $($OLLAMA_BIN --version 2>&1 | tail -1); mlx-lm $(python3 -c 'import mlx_lm; print(mlx_lm.__version__)')" | tee "$LOG/provenance.txt"
python3 -u scripts/bench_peer_prefill.py "$LOG/shortk.json" --backend metal --models 1.5B --depths 32,64,128,256 --n 6 2>&1 | tee "$LOG/shortk.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z'), bench_peer_prefill exit $rc" | tee -a "$LOG/provenance.txt"
exit $rc

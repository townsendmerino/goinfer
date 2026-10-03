#!/usr/bin/env bash
# Night job: Batch B's peer run, T1.11's 0.5B cells and T1.14 (docs/tasks/task-metal-audit-2026-10.md, "Batch B:
# pre-registration"), pre-registered there on 2026-10-02 before any graded run. One same-session bench_peer.py sweep
# on the Metal backend: goinfer (main as shipped, serve at REV), mlx-lm 0.31.3 (mlx-community 4-bit, its own quant) and
# Ollama, greedy, depth 128 (phase A) and 2048 and 3900 (phase B), 3 runs per cell, on the 0.5B, 1.5B and 7B. mlx has
# no 0.5B checkpoint here, so the 0.5B cells are goinfer and Ollama only (bench_peer skips the missing engine).
#
# Pinned: the serve binary and a scripts worktree at REV (the tree may move before tonight):
#   (cd ~/tmcode/goinfer/metal && CGO_ENABLED=0 go build -o ~/goinfer-bench/metal-audit-2026-10/serve-metal-main-<REV> ./cmd/serve)
#   git -C ~/tmcode/goinfer worktree add --detach ~/goinfer-bench/metal-audit-2026-10/wt-batch-b-<REV> <REV>
# Queued with:
#   python3 scripts/night.py add metal-audit-peer-b --est 90 --by "Claude (Mac session), Metal audit Batch B T1.11 + T1.14" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash ~/goinfer-bench/metal-audit-2026-10/run-peer-b.sh
# bench_peer.py holds the timing lock itself (inherited from the night runner). Results: ~/goinfer-logs/metal-audit-2026-10/batch-b/.
set -uo pipefail
REV=71812d57
BASE=$HOME/goinfer-bench/metal-audit-2026-10
WT=$BASE/wt-batch-b-$REV
LOG=$HOME/goinfer-logs/metal-audit-2026-10/batch-b
export GOINFER_SERVE_METAL=$BASE/serve-metal-main-$REV
export MLX_SERVER_BIN=$HOME/Library/Python/3.12/bin/mlx_lm.server
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_DEPTHS=2048,3900 BENCH_MODELS=0.5B,1.5B,7B
export BENCH_ENGINES=goinfer,mlx,ollama BENCH_RUNS=3 BENCH_SERVE_LOG_DIR=$LOG/serve-logs
mkdir -p "$LOG" "$LOG/serve-logs"
for f in "$GOINFER_SERVE_METAL" "$MLX_SERVER_BIN" "$OLLAMA_BIN"; do [ -x "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f "$WT/scripts/bench_peer.py" ] || { echo "missing scripts worktree $WT"; exit 1; }
cd "$WT"
echo "started: $(date '+%F %T %Z'); goinfer serve $GOINFER_SERVE_METAL (sha256 $(shasum -a 256 "$GOINFER_SERVE_METAL" | cut -c1-16)); ollama $($OLLAMA_BIN --version 2>&1 | tail -1); mlx-lm $(python3 -c 'import mlx_lm; print(mlx_lm.__version__)')" | tee "$LOG/peer-provenance.txt"
python3 -u scripts/bench_peer.py "$LOG/peer.json" 2>&1 | tee "$LOG/peer.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z'), bench_peer exit $rc" | tee -a "$LOG/peer-provenance.txt"
exit $rc

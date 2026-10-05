#!/usr/bin/env bash
# Night job: the peer read that refreshes docs/benchmarks.md's Metal decode row (docs/tasks/task-metal-audit-2026-10.md,
# "Peer refresh: pre-registration"), registered 2026-10-04 before it runs. TE5(a): the levers shipped since the
# post-merge read (B-P01, B-P02, D-B04, the serve chain, E-P05/E-P02/E-P08, int8) were graded new / old; the peer ratio is
# read once, after they ship, as its own same-session interleaved run, with no old arm. One bench_peer.py sweep on Metal:
# goinfer at REV, mlx-lm 0.31.3 and Ollama 0.32.5; greedy at depth 128 (phase A), 2048 and 3900 (phase B), and
# temp1.0_notrunc at depth 128 (phase C); 3 runs per cell; the 0.5B, 1.5B and 7B (no MLX 0.5B checkpoint here).
# Serve runs at its defaults (2 KV slots), as a user gets it.
#
# Pinned (the tree may move before tonight): serve binary ~/goinfer-bench/metal-audit-2026-10/serve-metal-peer-REV and
# scripts worktree ~/goinfer-bench/metal-audit-2026-10/wt-peer-REV (bench_peer.py, prompts.json).
# Queued with:
#   python3 scripts/night.py add metal-audit-peer-refresh --est 50 --priority 80 --by "Claude (Mac session), peer refresh" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash docs/measurements/metal-audit-2026-10/run-peer-refresh.sh
# Priority 80 runs it after tonight's serve-chain grade and M26 prefill grade (50 each). bench_peer.py holds the timing
# lock (inherited from the night runner) and uses the darwin instant idle gate.
set -uo pipefail
REV=a845af18
BASE=$HOME/goinfer-bench/metal-audit-2026-10
WT=$BASE/wt-peer-$REV
LOG=$HOME/goinfer-logs/metal-audit-2026-10/peer-refresh
export GOINFER_SERVE_METAL=$BASE/serve-metal-peer-$REV
export MLX_SERVER_BIN=$HOME/Library/Python/3.12/bin/mlx_lm.server
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_SAMPLED_BACKEND=metal BENCH_DEPTHS=2048,3900
export BENCH_CONFIGS=temp1.0_notrunc BENCH_MODELS=0.5B,1.5B,7B BENCH_ENGINES=goinfer,mlx,ollama BENCH_RUNS=3
export BENCH_SERVE_LOG_DIR=$LOG/serve-logs
mkdir -p "$LOG" "$LOG/serve-logs"
for f in "$GOINFER_SERVE_METAL" "$MLX_SERVER_BIN" "$OLLAMA_BIN"; do [ -x "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f "$WT/scripts/bench_peer.py" ] || { echo "missing scripts worktree $WT"; exit 1; }
cd "$WT"
echo "started: $(date '+%F %T %Z'); goinfer $REV (sha256 $(shasum -a 256 "$GOINFER_SERVE_METAL" | cut -c1-16)); ollama $($OLLAMA_BIN --version 2>&1 | tail -1); mlx-lm $(python3 -c 'import mlx_lm; print(mlx_lm.__version__)')" | tee "$LOG/provenance.txt"
python3 -u scripts/bench_peer.py "$LOG/peer.json" 2>&1 | tee "$LOG/peer.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z'), bench_peer exit $rc" | tee -a "$LOG/provenance.txt"
exit $rc

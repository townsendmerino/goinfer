#!/usr/bin/env bash
# Night job: the peer read after the Metal audit's merge (docs/tasks/task-metal-audit-2026-10.md, "Post-merge peer read:
# pre-registration"), registered 2026-10-03 before it runs. TE5(a): the levers were graded new / old with no peer arm;
# the peer ratio is read once, after they ship, as its own same-session interleaved run. One bench_peer.py sweep on
# Metal: goinfer at main c6f8100e (the merge), goinfer_old at main 71812d57 (last night's peer run, before the merge),
# mlx-lm 0.31.3 and Ollama; greedy at depth 128 (phase A) and 2048 and 3900 (phase B), and temp1.0_notrunc at depth 128
# (phase C); 3 runs per cell; the 0.5B, 1.5B and 7B (no mlx 0.5B here).
#
# Pinned (the tree may move before tonight):
#   serve binaries ~/goinfer-bench/metal-audit-2026-10/serve-metal-main-{c6f8100e,71812d57}
#   scripts worktree ~/goinfer-bench/metal-audit-2026-10/wt-peer-post-c6f8100e (bench_peer.py and prompts.json)
# Queued with:
#   python3 scripts/night.py add metal-audit-peer-post --est 60 --by "Claude (Mac session), post-merge peer read" \
#     --doc docs/tasks/task-metal-audit-2026-10.md -- bash docs/measurements/metal-audit-2026-10/run-peer-post-merge.sh
# bench_peer.py holds the timing lock (inherited from the night runner) and uses the darwin instant idle gate.
set -uo pipefail
NEW=c6f8100e
OLD=71812d57
BASE=$HOME/goinfer-bench/metal-audit-2026-10
WT=$BASE/wt-peer-post-$NEW
LOG=$HOME/goinfer-logs/metal-audit-2026-10/peer-post
export GOINFER_SERVE_METAL=$BASE/serve-metal-main-$NEW GOINFER_SERVE_METAL_OLD=$BASE/serve-metal-main-$OLD
export MLX_SERVER_BIN=$HOME/Library/Python/3.12/bin/mlx_lm.server
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_SAMPLED_BACKEND=metal BENCH_DEPTHS=2048,3900
export BENCH_CONFIGS=temp1.0_notrunc BENCH_MODELS=0.5B,1.5B,7B BENCH_ENGINES=goinfer,goinfer_old,mlx,ollama BENCH_RUNS=3
export BENCH_SERVE_LOG_DIR=$LOG/serve-logs
mkdir -p "$LOG" "$LOG/serve-logs"
for f in "$GOINFER_SERVE_METAL" "$GOINFER_SERVE_METAL_OLD" "$MLX_SERVER_BIN" "$OLLAMA_BIN"; do [ -x "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f "$WT/scripts/bench_peer.py" ] || { echo "missing scripts worktree $WT"; exit 1; }
cd "$WT"
echo "started: $(date '+%F %T %Z'); new $NEW (sha256 $(shasum -a 256 "$GOINFER_SERVE_METAL" | cut -c1-16)), old $OLD (sha256 $(shasum -a 256 "$GOINFER_SERVE_METAL_OLD" | cut -c1-16)); ollama $($OLLAMA_BIN --version 2>&1 | tail -1); mlx-lm $(python3 -c 'import mlx_lm; print(mlx_lm.__version__)')" | tee "$LOG/provenance.txt"
python3 -u scripts/bench_peer.py "$LOG/peer.json" 2>&1 | tee "$LOG/peer.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z'), bench_peer exit $rc" | tee -a "$LOG/provenance.txt"
exit $rc

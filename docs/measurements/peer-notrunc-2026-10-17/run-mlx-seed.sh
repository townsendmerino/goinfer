#!/bin/bash
# MLX server-path check (docs/measurements/peer-notrunc-2026-10-17.md, "MLX's two server paths"), a Mac night job (owner,
# 2026-10-17). mlx-lm only, Metal, depth 128, the 1.5B and 7B: greedy (now seeded, phase A), the shared preset with no seed
# (the diagnostic), and two seeded same-session controls (the preset, untruncated). bench_peer.py pinned at e90e82b9.
set -u
BASE=$HOME/goinfer-bench/peer-notrunc-2026-10-17
WT=$BASE/wt-scripts-e90e82b9
LOG=$HOME/goinfer-logs/peer-notrunc-2026-10-17/mlx-seed
export MLX_SERVER_BIN=$HOME/Library/Python/3.12/bin/mlx_lm.server
[ -x "$MLX_SERVER_BIN" ] || { echo "missing $MLX_SERVER_BIN"; exit 1; }
[ -f "$WT/scripts/bench_peer.py" ] || { echo "missing scripts worktree $WT"; exit 1; }
mkdir -p "$LOG" "$LOG/serve-logs"
cd "$WT" || exit 1
export GOINFER_SERVE_METAL=$BASE/serve-metal-63038121   # not run (BENCH_ENGINES=mlx); set so the plan resolves as the main job's did
export BENCH_ENGINES=mlx BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_SAMPLED_BACKEND=metal BENCH_MODELS=1.5B,7B
export BENCH_DEPTHS=none BENCH_RUNS=3 BENCH_SERVE_LOG_DIR=$LOG/serve-logs
export BENCH_CONFIGS=temp0.8_topk40_topp0.9_unseeded,temp0.8_topk40_topp0.9,temp1.0_notrunc
echo "started: $(date '+%F %T %Z'); bench_peer e90e82b9; mlx-lm $(python3 -c 'import mlx_lm; print(mlx_lm.__version__)'); MLX checkpoints Qwen2.5-Coder-1.5B-Instruct-4bit b3252a2f, Qwen2.5-7B-Instruct-4bit c26a38f6" | tee "$LOG/provenance.txt"
python3 -u scripts/bench_peer.py "$LOG/mlx-seed.json" 2>&1 | tee "$LOG/run.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z'), bench_peer exit $rc" | tee -a "$LOG/provenance.txt"
exit $rc

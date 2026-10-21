#!/usr/bin/env bash
# Night job (2026-10-18): the same-session greedy cell against seeded mlx-lm, registered in
# docs/measurements/peer-notrunc-2026-10-17.md, "Same-session greedy against MLX's single-request path". Metal, depth 128,
# 3 runs, goinfer and mlx-lm interleaved by bench_peer.py, the 1.5B and 7B, greedy only (phase A). bench_peer.py at
# e90e82b9 sends every arm seed 1 in greedy too, so mlx-lm runs its single-request path, and each MLX cell records it.
# The serve binary is the one the 2026-10-17 Metal run measured (63038121).
#
# The 7B is fit-checked at the job's start and measured only if it loads metal-resident. The guard is never bypassed.
set -uo pipefail
REV=63038121
BASE=$HOME/goinfer-bench/peer-notrunc-2026-10-17
WT=$BASE/wt-scripts-e90e82b9
LOG=$HOME/goinfer-logs/peer-notrunc-2026-10-17/mlx-greedy
export GOINFER_SERVE_METAL=$BASE/serve-metal-$REV
export MLX_SERVER_BIN=$HOME/Library/Python/3.12/bin/mlx_lm.server
export BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_SAMPLED_BACKEND=metal BENCH_DEPTHS=none
export BENCH_ENGINES=goinfer,mlx BENCH_RUNS=3
export BENCH_SERVE_LOG_DIR=$LOG/serve-logs
mkdir -p "$LOG" "$LOG/serve-logs"
for f in "$GOINFER_SERVE_METAL" "$MLX_SERVER_BIN"; do [ -x "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f "$WT/scripts/bench_peer.py" ] || { echo "missing scripts worktree $WT"; exit 1; }
cd "$WT"
python3 -c "
import sys; sys.path.insert(0, 'scripts'); import bench_peer as b
assert b.mlx_server_path(b.mlx_payload('m', 'p', b.CONFIGS['greedy'])) == 'single-request', 'greedy would run mlx-lm unseeded'" || exit 1
echo "started: $(date '+%F %T %Z'); goinfer $REV (sha256 $(shasum -a 256 "$GOINFER_SERVE_METAL" | cut -c1-16)); bench_peer e90e82b9; mlx-lm $(python3 -c 'import mlx_lm; print(mlx_lm.__version__)'); MLX checkpoints Qwen2.5-Coder-1.5B-Instruct-4bit b3252a2f, Qwen2.5-7B-Instruct-4bit c26a38f6; $(sysctl -n vm.swapusage)" | tee "$LOG/provenance.txt"

models=1.5B
"$GOINFER_SERVE_METAL" -model "bench=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf" -backend metal -addr 127.0.0.1:18972 > "$LOG/fitcheck-7B.log" 2>&1 &
pid=$!; for _ in $(seq 1 90); do grep -q "goinfer serving on" "$LOG/fitcheck-7B.log" && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
kill $pid 2>/dev/null; wait $pid 2>/dev/null
dp=$(grep -m1 'decode path' "$LOG/fitcheck-7B.log")
echo "7B fit check: $dp" | tee -a "$LOG/provenance.txt"
case "$dp" in *metal-resident*) models=1.5B,7B ;; *) echo "7B not measured: it would not load metal-resident tonight" | tee -a "$LOG/provenance.txt" ;; esac

BENCH_MODELS=$models python3 -u scripts/bench_peer.py "$LOG/mlx-greedy.json" 2>&1 | tee "$LOG/run.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z'), models $models, bench_peer exit $rc" | tee -a "$LOG/provenance.txt"
exit $rc

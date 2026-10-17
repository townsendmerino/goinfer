#!/usr/bin/env bash
# Night job (2026-10-17): the Metal no-truncation cells re-measured with temperature, top_p and top_k sent to every arm
# (scripts/bench_peer.py since 0600855e). Pre-registered in docs/measurements/peer-notrunc-2026-10-17.md, "Metal (the Mac,
# night of 2026-10-17)". The 2026-10-04 refresh's shape (metal-audit-2026-10/run-peer-refresh.sh): Metal, depth 128,
# 3 runs, goinfer / mlx-lm / Ollama, the 0.5B, 1.5B and 7B (no MLX 0.5B checkpoint); bench_peer.py adds its greedy cells at
# 128 as same-session controls. bench_peer.py holds the timing lock (inherited from the night runner) and uses the darwin
# instant idle gate.
#
# The 7B's Metal resident build was over the memory guard's live budget at the by-day fit check (4.65 GB against
# 4.43 GB): it is checked again here, and measured only if it loads metal-resident. The guard is never bypassed.
#
# Pinned: serve binary and scripts worktree under ~/goinfer-bench/peer-notrunc-2026-10-17, at REV.
set -uo pipefail
REV=63038121
BASE=$HOME/goinfer-bench/peer-notrunc-2026-10-17
WT=$BASE/wt-$REV
LOG=$HOME/goinfer-logs/peer-notrunc-2026-10-17
export GOINFER_SERVE_METAL=$BASE/serve-metal-$REV
export MLX_SERVER_BIN=$HOME/Library/Python/3.12/bin/mlx_lm.server
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_SAMPLED_BACKEND=metal BENCH_DEPTHS=none
export BENCH_CONFIGS=temp1.0_notrunc BENCH_ENGINES=goinfer,mlx,ollama BENCH_RUNS=3
export BENCH_SERVE_LOG_DIR=$LOG/serve-logs
mkdir -p "$LOG" "$LOG/serve-logs"
for f in "$GOINFER_SERVE_METAL" "$MLX_SERVER_BIN" "$OLLAMA_BIN"; do [ -x "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f "$WT/scripts/bench_peer.py" ] || { echo "missing scripts worktree $WT"; exit 1; }
cd "$WT"
python3 -c "
import sys; sys.path.insert(0, 'scripts'); import bench_peer as b
c = b.CONFIGS['temp1.0_notrunc']
assert all(k in c[e] for e in ('goinfer', 'ollama', 'mlx') for k in ('temperature', 'top_p', 'top_k')), c" || exit 1
echo "started: $(date '+%F %T %Z'); goinfer $REV (sha256 $(shasum -a 256 "$GOINFER_SERVE_METAL" | cut -c1-16)); ollama $($OLLAMA_BIN --version 2>&1 | tail -1); mlx-lm $(python3 -c 'import mlx_lm; print(mlx_lm.__version__)')" | tee "$LOG/provenance.txt"

models=0.5B,1.5B
"$GOINFER_SERVE_METAL" -model "bench=$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf" -backend metal -addr 127.0.0.1:18971 > "$LOG/fitcheck-7B.log" 2>&1 &
pid=$!; for _ in $(seq 1 90); do grep -q "goinfer serving on" "$LOG/fitcheck-7B.log" && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
kill $pid 2>/dev/null; wait $pid 2>/dev/null
dp=$(grep -m1 'decode path' "$LOG/fitcheck-7B.log")
echo "7B fit check: $dp" | tee -a "$LOG/provenance.txt"
case "$dp" in *metal-resident*) models=0.5B,1.5B,7B ;; *) echo "7B not measured: it would not load metal-resident tonight" | tee -a "$LOG/provenance.txt" ;; esac

BENCH_MODELS=$models python3 -u scripts/bench_peer.py "$LOG/peer.json" 2>&1 | tee "$LOG/peer.log"
rc=${PIPESTATUS[0]}
echo "finished: $(date '+%F %T %Z'), models $models, bench_peer exit $rc" | tee -a "$LOG/provenance.txt"
exit $rc

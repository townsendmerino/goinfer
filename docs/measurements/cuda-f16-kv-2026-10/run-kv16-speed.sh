#!/usr/bin/env bash
# The paired speed check of docs/tasks/task-cuda-f16-kv-2026-10.md as an unattended night job: ONE serve binary built from REV, two arms -- "goinfer" runs it with
# -kv f16 and "goinfer_old" runs the SAME binary with -kv f32 (BENCH_GOINFER_ARGS / BENCH_GOINFER_OLD_ARGS) -- interleaved per cell by scripts/bench_peer.py, no peer arm.
# Cells: {0.5B, 1.5B, phi3-mini} x {128, 512, 2048, 3900}, and 7B (the D7 model) at 128 and 8000 with the context pinned to 8448. Five runs per cell. The whole sweep is run
# twice with the arm order reversed, so a first-in-cell effect cannot read as a speed difference. bench_peer.py holds the timing lock for its whole run.
#
# DRY=1 stops after the worktree, the build and the provenance block (the unattended-script check).
# Estimated 70 minutes: smoke cell 8 s plus 3 s of server start and teardown on the 0.5B; 24 small cells x about 40 s, the 7B cells (load ~30 s, 8000-key decode ~20 s a run) about
# 10 minutes a pass, twice. Grade with python3 docs/measurements/cuda-f16-kv-2026-10/grade-speed.py <outdir>.
#   python3 scripts/night.py add kv16-speed --est 70 --by "<who>, CUDA f16 KV" --doc docs/tasks/task-cuda-f16-kv-2026-10.md -- env REV=<sha> bash docs/measurements/cuda-f16-kv-2026-10/run-kv16-speed.sh
set -uo pipefail
REV=${REV:?REV is required: the commit under test, never a default}
SRC=$HOME/mycode/goinfer
AIKIT=${AIKIT:-$HOME/mycode/aikit/aikit}
BASE=${BASE:-$HOME/goinfer-bench/kv16-speed}
OUT=${OUT:-$HOME/goinfer-logs/kv16/speed-$(date +%F)}
export PATH=/usr/local/go/bin:$PATH
case "$BASE $OUT" in *"/srv/models"*|*"/Volumes/"*) echo "FATAL: an archive path"; exit 2;; esac
for p in "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf" "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" "$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf" "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"; do
  [ -e "$p" ] || { echo "FATAL: $p is missing"; exit 2; }
done
mkdir -p "$BASE" "$OUT"
cd "$SRC" || exit 2
WT=$BASE/wt
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
cd "$WT" || exit 2
printf 'go 1.27.0\n\ntoolchain go1.27.2\n\nuse (\n\t.\n\t./cuda\n\t./gpu\n\t%s\n)\n' "$AIKIT" > go.work
BIN=$BASE/serve-cuda-$(git rev-parse --short=8 HEAD)
(cd cuda && CGO_ENABLED=0 go build -tags cuda -o "$BIN" ./cmd/serve) || { echo "FATAL: serve build failed"; exit 2; }
{ echo "rev:      $(git rev-parse HEAD)"; echo "aikit:    $(git -C "$AIKIT" rev-parse HEAD)"; echo "binary:   $BIN"; echo "started:  $(date '+%F %T %Z')"
  echo "driver:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"; echo "arms:     goinfer = -kv f16, goinfer_old = -kv f32 (one binary)"; } | tee "$OUT/provenance.txt"
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after the build and the provenance block"; cd "$SRC" && git worktree remove --force "$WT"; exit 0; }
export GOINFER_SERVE_CUDA=$BIN GOINFER_SERVE_CUDA_OLD=$BIN BENCH_BACKENDS=cuda BENCH_RUNS=5
export BENCH_GOINFER_ARGS="-kv f16" BENCH_GOINFER_OLD_ARGS="-kv f32"
rc=0
pass() { # pass <name> <engines> <models> <depths> [extra env...]
  local name=$1 eng=$2 models=$3 depths=$4; shift 4
  echo "== $name: start $(date '+%T')"
  env "$@" BENCH_ENGINES=$eng BENCH_MODELS=$models BENCH_DEPTHS=$depths BENCH_SERVE_LOG_DIR="$OUT/$name-serve-logs" \
    python3 -u scripts/bench_peer.py "$OUT/$name.json" > "$OUT/$name.log" 2>&1
  local r=$?
  echo "== $name: exit $r $(date '+%T')"
  [ $r -eq 0 ] && [ -s "$OUT/$name.json" ] || { echo "!! $name exited $r or wrote no results (bench_peer exits 0 when it refuses a busy box)"; rc=1; }
}
pass small-f16first goinfer,goinfer_old 0.5B,1.5B,phi3-mini 512,2048,3900 X=1
pass small-f32first goinfer_old,goinfer 0.5B,1.5B,phi3-mini 512,2048,3900 X=1
pass d7-f16first goinfer,goinfer_old 7B 8000 BENCH_CTX=8448
pass d7-f32first goinfer_old,goinfer 7B 8000 BENCH_CTX=8448
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc

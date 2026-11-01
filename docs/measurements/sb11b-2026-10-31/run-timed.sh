#!/usr/bin/env bash
# SB11b's timed gates (docs/tasks/task-subsystems-2026-10.md §8, "SB11b"), on nobara's RTX 2070 SUPER, Qwen2.5-Coder-0.5B
# q4_k_m:
#   G-SB11b-5: this build's served decode (timings on, cost off) against main before it, bench_peer.py with
#              BENCH_ENGINES=goinfer,goinfer_old, phase A, two passes with the engine order reversed (A B B A).
#   G-SB11b-6: one server of this build, the same prompt, 8 requests alternating cost off and on.
# Usage: run-timed.sh <new source snapshot> <old source snapshot> <out dir>. Each snapshot has its own go.work. The builds
# hold the timing lock shared; bench_peer.py takes it exclusive; G-SB11b-6 runs under the exclusive lock.
set -u
NEW=$1 OLD=$2 OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
mkdir -p "$OUT/bin"
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/run.log"; }
tl() { python3 "$REPO/scripts/timing_lock.py" "$@"; }

log "build: new $(cat "$NEW/SNAPSHOT_REV"), old $(cat "$OLD/SNAPSHOT_REV")"
tl run --shared --wait 3600 --label 'nobara-mm01 SB11b: build two serve-cuda (~3 min)' -- bash -c "
  (cd '$NEW/cuda' && GOWORK='$NEW/go.work' CGO_ENABLED=0 go build -tags cuda -o '$OUT/bin/serve-new' ./cmd/serve) &&
  (cd '$OLD/cuda' && GOWORK='$OLD/go.work' CGO_ENABLED=0 go build -tags cuda -o '$OUT/bin/serve-old' ./cmd/serve)" >> "$OUT/build.log" 2>&1
rc=$?; log "build exit $rc"; [ $rc -eq 0 ] || exit 2
sha256sum "$OUT/bin/serve-new" "$OUT/bin/serve-old" | tee -a "$OUT/run.log"

i=0
for order in goinfer,goinfer_old goinfer_old,goinfer; do
  i=$((i+1))
  log "G-SB11b-5 pass $i: engines $order"
  GOINFER_SERVE_CUDA="$OUT/bin/serve-new" GOINFER_SERVE_CUDA_OLD="$OUT/bin/serve-old" BENCH_ENGINES=$order BENCH_BACKENDS=cuda \
    BENCH_MODELS=0.5B BENCH_DEPTHS=none BENCH_PHASES=A BENCH_IDLE_WAIT=1800 \
    python3 "$REPO/scripts/bench_peer.py" "$OUT/pass$i.json" > "$OUT/pass$i.log" 2>&1
  rc=$?; log "pass $i exit $rc"
  [ -s "$OUT/pass$i.json" ] || { log "VOID pass $i: no results file (a refusal exits 0 without one)"; exit 3; }
done
python3 -I "$HERE/grade_ab.py" "$OUT/pass1.json" "$OUT/pass2.json" 2>&1 | tee -a "$OUT/run.log"

log "G-SB11b-6: cost on against off"
tl run --wait 3600 --label 'nobara-mm01 SB11b G-SB11b-6: cost on/off (~3 min)' -- \
  python3 -I "$HERE/cost_on.py" "$OUT/bin/serve-new" "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf" "$OUT/cost_on.json" 2>&1 | tee -a "$OUT/run.log"
log "end"

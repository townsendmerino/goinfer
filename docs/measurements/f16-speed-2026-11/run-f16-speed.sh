#!/usr/bin/env bash
# F16-speed (docs/measurements/f16-speed-2026-11/registration.md) as a nobara night job. ONE serve binary built from REV, two arms: "goinfer" runs
# it with -kv f16 and "goinfer_old" runs the SAME binary with -kv f32, interleaved per cell by scripts/bench_peer.py, no peer arm.
# Cells: {0.5B, 1.5B, phi3-mini} x {512, 2048, 3900}, and 7B at 8000 with the context pinned to 8448. Five runs per cell. The sweep runs twice, the
# arm order reversed, so a first-in-cell effect cannot read as a speed difference. bench_peer.py holds the timing lock for each pass.
# After each pass every serve log must name the cuda-resident decode path and the arm's KV as delivered ("KV f16 (requested f16", "KV f32
# (requested f32"); a log that does not makes the pass void (rc 1), whatever its rates say.
#
# DRY=1 stops after the worktree, the build and the provenance block (the unattended-script check).
# Estimated 70 minutes: about 22 a pass (18 small arm-cells at about 40 s, the 7B's two at about 5 min with its load), twice, plus the build and
# the idle waits. Grade: python3 docs/measurements/cuda-f16-kv-2026-10/grade-speed.py <outdir>.
# Queue it from a copy taken out of <sha> (registration.md, "Night recipe"), so the checkout's working tree is not touched.
set -uo pipefail
REV=${REV:?REV is required: the commit under test, never a default}
SRC=${SRC:-$HOME/mycode/goinfer}
BASE=${BASE:-$HOME/goinfer-bench/f16-speed}
OUT=${OUT:-$HOME/goinfer-logs/f16-speed/speed-$(date +%F)}
export PATH=/usr/local/go/bin:$PATH
case "$BASE $OUT" in *"/srv/models"*|*"/Volumes/"*) echo "FATAL: an archive path"; exit 2;; esac
for p in "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf" "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" "$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf" "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"; do
  [ -e "$p" ] || { echo "FATAL: $p is missing"; exit 2; }
done
mkdir -p "$BASE" "$OUT"
cd "$SRC" || exit 2
git fetch -q origin || { echo "FATAL: git fetch failed"; exit 2; }
WT=$BASE/wt
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
cd "$WT" || exit 2
go work init . ./backend/cuda || { echo "FATAL: go work init failed"; exit 2; }
BIN=$BASE/serve-cuda-$(git rev-parse --short=9 HEAD)
(cd backend/cuda && GOWORK=$WT/go.work CGO_ENABLED=0 go build -tags cuda -o "$BIN" ./cmd/serve) || { echo "FATAL: serve build failed"; exit 2; }
{ echo "rev:      $(git rev-parse HEAD)"; echo "aikit:    $(awk '$1 == "github.com/townsendmerino/aikit" {print $2}' go.mod)"; echo "binary:   $BIN"
  echo "started:  $(date '+%F %T %Z')"; echo "driver:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
  echo "arms:     goinfer = -kv f16, goinfer_old = -kv f32 (one binary)"; } | tee "$OUT/provenance.txt"
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stopping after the build and the provenance block"; cd "$SRC" && git worktree remove --force "$WT"; exit 0; }
export GOINFER_SERVE_CUDA=$BIN GOINFER_SERVE_CUDA_OLD=$BIN BENCH_BACKENDS=cuda BENCH_RUNS=5
export BENCH_GOINFER_ARGS="-kv f16" BENCH_GOINFER_OLD_ARGS="-kv f32"
rc=0
# bench_peer.py refuses to start above a 1-minute load of 1.0, and a previous night job's load decays slowly: wait for the box to settle before
# each pass; bench_peer's own refusal still stands if it does not.
settle() {
  for _ in $(seq 1 90); do
    awk '{exit !($1 < 0.7)}' /proc/loadavg && return 0
    sleep 10
  done
  echo "!! load still $(cut -d' ' -f1 /proc/loadavg) after 15 min; bench_peer will decide"
}
delivered() { # delivered <serve-log dir>: every arm's log names the resident path and its KV as delivered
  local d=$1 f want n=0 bad=0
  for f in "$d"/goinfer-cuda-*.log "$d"/goinfer_old-cuda-*.log; do
    [ -e "$f" ] || continue
    n=$((n + 1))
    case $(basename "$f") in goinfer_old-*) want="KV f32 (requested f32" ;; *) want="KV f16 (requested f16" ;; esac
    grep -q "decode path: cuda-resident" "$f" && grep -qF "$want" "$f" || { echo "!! $f: not cuda-resident, or not '$want'"; bad=1; }
  done
  [ $n -gt 0 ] && [ $bad -eq 0 ]
}
pass() { # pass <name> <engines> <models> <depths> [extra env...]
  local name=$1 eng=$2 models=$3 depths=$4; shift 4
  settle
  echo "== $name: start $(date '+%T')"
  env "$@" BENCH_ENGINES=$eng BENCH_MODELS=$models BENCH_DEPTHS=$depths BENCH_SERVE_LOG_DIR="$OUT/$name-serve-logs" \
    python3 -u scripts/bench_peer.py "$OUT/$name.json" > "$OUT/$name.log" 2>&1
  local r=$?
  echo "== $name: exit $r $(date '+%T')"
  [ $r -eq 0 ] && [ -s "$OUT/$name.json" ] || { echo "!! $name exited $r or wrote no results (bench_peer exits 0 when it refuses a busy box)"; rc=1; }
  delivered "$OUT/$name-serve-logs" || { echo "!! $name: VOID, a serve log does not show the arm's KV delivered on the resident path"; rc=1; }
}
pass small-f16first goinfer,goinfer_old 0.5B,1.5B,phi3-mini 512,2048,3900
pass small-f32first goinfer_old,goinfer 0.5B,1.5B,phi3-mini 512,2048,3900
pass d7-f16first goinfer,goinfer_old 7B 8000 BENCH_CTX=8448
pass d7-f32first goinfer_old,goinfer 7B 8000 BENCH_CTX=8448
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc

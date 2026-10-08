#!/usr/bin/env bash
# Triage of the first heavy-tier gate's two untriaged failures (docs/tasks/task-multimodal-support-2026-10.md, "Build-scratch / margin accounting" and the TestSpecNonCopyLane record after it).
#
#  1. The WebGPU hang: `go run ./cmd/gate gpu`'s webgpu-parity cell sat on TestRMSNormBatched_parity (the 65th test, started ~20 s in) for more than 10 minutes until the cell's timeout. For BOTH revisions
#     (BASEREV, the baseline the other four pre-existing failures were shown at, and HEADREV) this runs:
#       alone  the one test by itself             -> hangs alone, or only after the 64 before it?
#       cell   the gate's own -run pattern, whole -> does the hang reproduce, and where do the goroutines sit?
#     each under `go test -timeout 4m`, which on a hang prints every goroutine's stack. Reading: hang at both revs = pre-existing (not ours to fix tonight; the stack names the call);
#     hang at HEADREV only = a regression between the two, bisect next; no hang anywhere = it needs the gate's heavier surroundings (the CUDA tier running before it) and the next step is that.
#  2. TestSpecNonCopyLane, at HEADREV only: its failure was root-caused by tokenizing (the live benchmarks.md slice outgrew the pinned context) and fixed by a frozen document; this is the first run of
#     the fixed test, a measurement that prints its own numbers. Nothing is graded here: a pass is the test's own "--- PASS" line, a skip or a missing line is a failure of this job.
#
# Each revision is tested in its own detached worktree (the tree may move before tonight); the gitignored fixtures are symlinked in from the main checkout, as the gate's night script does.
# With TRIAGE_SKIP_WEBGPU=1 only the TestSpecNonCopyLane step runs (~15-20 min). Full estimate: ~45 min (setup 2 x ~2 min; the four WebGPU steps <= 4 min each, ~16 min worst; TestSpecNonCopyLane ~10-15 min, -timeout 30m).
#   python3 scripts/night.py add gate-triage-webgpu-noncopy --est 45 --by "nobara session, gate triage" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/margin/run-gate-triage.sh
#   TRIAGE_DRY=1 bash .../run-gate-triage.sh /tmp/somewhere     # plumbing control: worktrees, go.work, compile both test binaries, run nothing
#   TRIAGE_CONTROL=1 bash .../run-gate-triage.sh /tmp/somewhere # the step mechanism on ONE real step (webgpu-alone-head, -timeout 1m)
# Writes $LOG/triage-summary.txt: one line per step (rc, wall seconds, verdict word), then the first stack frames of any timed-out step.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../../.." && pwd)
HEADREV=${HEADREV:-a3d03268b7b312ee8f51b59f7ba1e86ca5871584}
BASEREV=${BASEREV:-12c85f4a}
LOG=${1:-$HOME/goinfer-logs/gate-triage-$(date +%F)}
WTDIR=${WTDIR:-$HOME/goinfer-bench/gate-triage}
RUNPAT='Parity|_parity|matchesCPU|matchesSequential' # cmd/gate/parity.go webgpuParityRun
export PATH=/usr/local/go/bin:$PATH
[ -d "$HOME/models" ] || { echo "FATAL: $HOME/models is missing"; exit 2; }
mkdir -p "$LOG" "$WTDIR"
SUM="$LOG/triage-summary.txt"; : > "$SUM"
cd "$SRC" || exit 2
for r in "$BASEREV" "$HEADREV"; do git cat-file -e "$r^{commit}" 2>/dev/null || { echo "FATAL: $r is not a commit here"; exit 2; }; done

# mkwt NAME REV: a detached worktree of REV at $WTDIR/NAME with the main checkout's untracked/ignored testdata linked in and a go.work over the five modules' three that matter.
mkwt() {
  local name=$1 rev=$2 wt="$WTDIR/$1" n=0 d p c
  [ -d "$wt" ] && git worktree remove --force "$wt"
  git worktree add -q --detach "$wt" "$rev" || { echo "FATAL: cannot check out $rev"; return 1; }
  for d in testdata decoder/testdata cuda/testdata gpu/testdata; do
    [ -d "$SRC/$d" ] || continue
    while IFS= read -r p; do
      case "$p" in */) p=${p%/} ;; *) ;; esac
      [ -e "$wt/$p" ] && continue
      mkdir -p "$(dirname "$wt/$p")"; ln -s "$SRC/$p" "$wt/$p"; n=$((n + 1))
    done < <(git ls-files --others --exclude-standard --directory "$d"; git ls-files --others --ignored --exclude-standard --directory "$d")
  done
  ( cd "$wt" && go work init . ./gpu ./cuda ./metal ./demo/agent ) || { echo "FATAL: go work init in $wt"; return 1; }
  echo "$name: $(git -C "$wt" rev-parse --short HEAD), $n fixture entries linked"
}

# step NAME DIR TIMEOUT_NOTE CMD...: run CMD in DIR with output to $LOG/NAME.log, a heartbeat every 60 s, and one summary line. Never aborts the script.
step() {
  local name=$1 dir=$2; shift 2
  local t0=$SECONDS rc
  echo "[$(date +%T)] START $name"
  ( while sleep 60; do echo "[$(date +%T)] ... $name running $((SECONDS - t0))s; log $(wc -c < "$LOG/$name.log") bytes, last: $(tail -n1 "$LOG/$name.log" | cut -c1-110)"; done ) &
  local tick=$!
  ( cd "$dir" && "$@" ) > "$LOG/$name.log" 2>&1
  rc=$?
  kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
  local verdict=ok
  if grep -q "panic: test timed out" "$LOG/$name.log"; then verdict=TIMED-OUT
  elif [ "$rc" -ne 0 ]; then verdict="FAIL(rc=$rc)"; fi
  printf '%-28s rc=%-3s %5ss  %s\n' "$name" "$rc" "$((SECONDS - t0))" "$verdict" | tee -a "$SUM"
}

summarize() {
  # A pass is the test's own PASS line. A skip (no model) or no line at all is a failure of this job, whatever go test's exit code was.
  if [ -e "$LOG/noncopy-head.log" ] && ! grep -q -- '--- PASS: TestSpecNonCopyLane' "$LOG/noncopy-head.log"; then echo "!! TestSpecNonCopyLane did not print its PASS line (skipped, failed or timed out): see $LOG/noncopy-head.log" | tee -a "$SUM"; fi
  for f in "$LOG"/webgpu-*.log; do
    if grep -q "panic: test timed out" "$f"; then
      { echo; echo "== $(basename "$f"): timed out. In flight at the timeout, then the first stack frames:"
        grep -E "^(=== RUN|=== CONT)" "$f" | tail -1
        grep -n -A40 -m1 "panic: test timed out" "$f" | cut -c1-170 | head -60; } | tee -a "$SUM"
    fi
  done
}

{ echo "head: $HEADREV"; echo "base: $BASEREV"; echo "started: $(date '+%F %T %Z')"; echo "go: $(go version)"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load: $(cat /proc/loadavg)"; } | tee "$LOG/provenance.txt"

# TRIAGE_SKIP_WEBGPU=1: only the fixed TestSpecNonCopyLane runs (the WebGPU hang was root-caused and fixed by day on 2026-10-08, gpu.TestContextClose_finalizerRace; its steps would repeat a closed question).
[ -z "${TRIAGE_SKIP_WEBGPU:-}" ] && { mkwt base "$BASEREV" || exit 2; }
mkwt head "$HEADREV" || exit 2

if [ -n "${TRIAGE_DRY:-}" ]; then
  for w in base head; do
    [ -d "$WTDIR/$w" ] || continue
    step "dry-compile-gpu-$w" "$WTDIR/$w" go test -c -o /dev/null -tags 'gpu goinfer_testhooks' ./gpu/
  done
  step "dry-compile-cuda-head" "$WTDIR/head/cuda" go test -c -o /dev/null -tags 'cuda goinfer_testhooks' .
  echo "dry run finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"; exit 0
fi

export GOINFER_DNET_PARITY=1 GOINFER_SSM_PARITY=1 # the webgpu-parity cell's env (cmd/gate/gpu.go)
if [ -n "${TRIAGE_CONTROL:-}" ]; then # the real step mechanism (heartbeat, timeout parse, summary) on one bounded step, by day: ~1 min
  step "webgpu-alone-head" "$WTDIR/head" go test -count=1 -tags 'gpu goinfer_testhooks' -run '^TestRMSNormBatched_parity$' -v -timeout "${TRIAGE_CONTROL_TIMEOUT:-1m}" ./gpu/
  summarize; echo "control finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"; exit 0
fi
if [ -z "${TRIAGE_SKIP_WEBGPU:-}" ]; then
  for w in base head; do
    step "webgpu-alone-$w" "$WTDIR/$w" go test -count=1 -tags 'gpu goinfer_testhooks' -run '^TestRMSNormBatched_parity$' -v -timeout 4m ./gpu/
    step "webgpu-cell-$w"  "$WTDIR/$w" go test -count=1 -tags 'gpu goinfer_testhooks' -run "$RUNPAT" -v -timeout 4m ./gpu/
  done
fi
step "noncopy-head" "$WTDIR/head/cuda" env GOINFER_HEAVY_TESTS=1 go test -count=1 -tags 'cuda goinfer_testhooks' -run '^TestSpecNonCopyLane$' -v -timeout 30m .

summarize
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
for w in base head; do [ -d "$WTDIR/$w" ] && git -C "$SRC" worktree remove --force "$WTDIR/$w" 2>/dev/null; done
exit 0

#!/usr/bin/env bash
# Device validation of the branch `gpu-explicit-release` (commit REV, local, NOT on main): a wrapper closed after its Context releases on the finalizer goroutine (Context.releaseOwned), the live-buffer
# gauge covers every resident weight type, and the test run reports which tests close a Context with device buffers still live. Written 2026-10-08 as the follow-up to b770a47c's "not covered" list
# (docs/tasks/task-multimodal-support-2026-10.md, "The WebGPU parity-cell hang"). Nothing here was run on a device when the branch was committed: only TestReleaseOwned (no device) had.
#
#   reproducers  TestContextClose_* + TestReleaseOwned           -> both reproducers pass (each loops 150 rounds under a 90 s deadline)
#   suite        the whole ./gpu/ package, -v                    -> 142 pass / 0 fail as at b770a47c (57 skips are fixture-gated); the LEAK REPORT at the end lists tests to fix
#   cell-loop    the webgpu-parity cell, 10 runs of a built test binary, 2 min each -> all clean (before b770a47c: 1 hang in 8)
#   planted      TestContextClose_lateWrapperClose_plantedDefect (the hop OFF) -> must be RED (deadlock by design); GREEN means the reproducer cannot see the defect
# Estimate: ~20 min (reproducers ~3, suite ~4, cell loop ~5.5, planted <=2, setup ~2).
#   python3 scripts/night.py add gpu-release-validate --est 20 --by "nobara session, WebGPU follow-up" --doc docs/tasks/task-multimodal-support-2026-10.md -- bash docs/measurements/multimodal-support-2026-10/margin/run-gpu-release-validate.sh
# PLUMBING NOT CONTROLLED: this script was queued while the night queue held the box, so it has never run (bash -n only). Its worktree, fixture and step pieces are copied from run-gate-triage.sh, which was.
# Writes $LOG/validate-summary.txt. A step that did not print its own verdict is a failure of this job, whatever the exit code.
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../../../.." && pwd)
REV=${REV:-bf77ca502fcd59a6979dff96f44864d5e3104c0f}
LOG=${1:-$HOME/goinfer-logs/gpu-release-validate-$(date +%F)}
WT=${WT:-$HOME/goinfer-bench/gpu-release-validate/wt}
export PATH=/usr/local/go/bin:$PATH
mkdir -p "$LOG" "$(dirname "$WT")"
SUM="$LOG/validate-summary.txt"; : > "$SUM"
cd "$SRC" || exit 2
git cat-file -e "$REV^{commit}" 2>/dev/null || { echo "FATAL: $REV is not a commit here (the branch gpu-explicit-release is local)"; exit 2; }
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
n=0
for d in testdata decoder/testdata cuda/testdata gpu/testdata; do
  [ -d "$SRC/$d" ] || continue
  while IFS= read -r p; do
    p=${p%/}; [ -e "$WT/$p" ] && continue
    mkdir -p "$(dirname "$WT/$p")"; ln -s "$SRC/$p" "$WT/$p"; n=$((n + 1))
  done < <(git ls-files --others --exclude-standard --directory "$d"; git ls-files --others --ignored --exclude-standard --directory "$d")
done
( cd "$WT" && go work init . ./gpu ./cuda ./metal ./demo/agent ) || { echo "FATAL: go work init"; exit 2; }
{ echo "rev: $(git -C "$WT" rev-parse HEAD)"; echo "started: $(date '+%F %T %Z')"; echo "fixtures: $n linked"; echo "go: $(go version)"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader)"; echo "load: $(cat /proc/loadavg)"; } | tee "$LOG/provenance.txt"

TAGS='gpu goinfer_testhooks'
step() { # NAME CMD... : output to $LOG/NAME.log, heartbeat every 60 s, one summary line
  local name=$1; shift
  local t0=$SECONDS rc
  echo "[$(date +%T)] START $name"
  ( while sleep 60; do echo "[$(date +%T)] ... $name running $((SECONDS - t0))s, log $(wc -c < "$LOG/$name.log") bytes"; done ) &
  local tick=$!
  ( cd "$WT/gpu" && "$@" ) > "$LOG/$name.log" 2>&1
  rc=$?
  kill "$tick" 2>/dev/null; wait "$tick" 2>/dev/null
  local v=ok; grep -q "panic: test timed out" "$LOG/$name.log" && v=TIMED-OUT; [ "$rc" -ne 0 ] && [ "$v" = ok ] && v="FAIL(rc=$rc)"
  printf '%-28s rc=%-3s %5ss  %s\n' "$name" "$rc" "$((SECONDS - t0))" "$v" | tee -a "$SUM"
}
export GOINFER_DNET_PARITY=1 GOINFER_SSM_PARITY=1

step reproducers go test -count=1 -tags "$TAGS" -run 'TestContextClose_finalizerRace|TestContextClose_lateWrapperClose$|TestReleaseOwned' -v -timeout 6m .
for t in TestContextClose_finalizerRace TestContextClose_lateWrapperClose TestReleaseOwned; do
  grep -q -- "--- PASS: $t " "$LOG/reproducers.log" || echo "!! $t did not print its PASS line (skipped, failed or timed out)" | tee -a "$SUM"
done

step suite go test -count=1 -tags "$TAGS" -v -timeout 14m .
echo "suite: pass=$(grep -c '^--- PASS' "$LOG/suite.log") skip=$(grep -c '^--- SKIP' "$LOG/suite.log") fail=$(grep -c '^--- FAIL' "$LOG/suite.log")" | tee -a "$SUM"
if grep -q "LEAK REPORT" "$LOG/suite.log"; then { echo "== leak report (tests that closed a Context with device buffers still live):"; sed -n '/LEAK REPORT/,$p' "$LOG/suite.log" | head -60; } | tee -a "$SUM"; else echo "== leak report: none" | tee -a "$SUM"; fi

( cd "$WT/gpu" && go test -c -o "$LOG/gpu.test" -tags "$TAGS" . ) || echo "!! cell-loop binary did not build" | tee -a "$SUM"
for i in 1 2 3 4 5 6 7 8 9 10; do
  step "cell-$i" "$LOG/gpu.test" -test.run 'Parity|_parity|matchesCPU|matchesSequential' -test.v -test.timeout 2m
done
echo "cell loop: cell-1..cell-10 above must each read ok (65 pass / 8 skip each; before b770a47c one run in eight hung)" >> "$SUM"

GOINFER_GPU_PLANTED=1 step planted go test -count=1 -tags "$TAGS" -run 'TestContextClose_lateWrapperClose_plantedDefect' -v -timeout 3m .
if grep -qE "deadlock|panic: test timed out" "$LOG/planted.log"; then echo "planted defect: RED as intended" | tee -a "$SUM"; else echo "!! planted defect: NOT detected (the reproducer cannot see it, or it skipped)" | tee -a "$SUM"; fi

echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"
git -C "$SRC" worktree remove --force "$WT" 2>/dev/null
exit 0

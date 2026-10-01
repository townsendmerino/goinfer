#!/usr/bin/env bash
# Bisect for the v0.20.0 sweep's one real red: TestQwen35GGUF_gate measured argmax 57/80 against its 66 floor on
# nobara-pc (sweep run 2, bcf50a49, 2026-10-01), and passed in the v0.19.0 sweep on the same box. All five
# divergences are rank-2 near-ties (gap 0.0002-0.0072 of the logit range), and the cosines did not drop (min 0.98786,
# mean 0.99619, against 0.98740 / 0.99608 at the 2026-08-22 re-baseline). So the question is which commit moved the
# near-ties, and whether it meant to.
#
# Pre-registered before the run (owner OK 2026-10-01 to run it by day, once the sweep ends):
# - A step's verdict comes from the gate's own summary line: GOOD when argmax >= 66/80 (the test's floor, unchanged
#   across v0.19.0..bcf50a49), BAD below it, SKIP (125) when the line is missing (a build failure or a load error).
#   The test's other bars are recorded, not graded.
# - GOOD (v0.19.0) is measured first. If it does not test good on today's box and toolchain, the bisect stops there,
#   because it would be bisecting nothing.
# - Every tested commit's summary line is kept (progress.log), since a near-tie can move at more than one commit.
# - The answer is the first bad commit. If it is a deliberate numerics change, the floor is re-set only with that
#   mechanism recorded and the owner's OK. Anything else is a bug, and gets fixed.
#
# Range: commits touching decoder/, the root-module packages it imports (constrain, internal/giw, tokenizer) and
# go.mod/go.sum (the aikit pin). 164 of them, so about 8 steps after the GOOD check. A step is a build plus ~5 min of
# test (299 s in the sweep): about 9 x 6-7 min, ~60 min in all.
#
# GOINFER_NO_FIT_GUARD=1 comes from the environment: the guard refuses this load in part of the range, and the variable
# is read at Load at every commit in it (os.Getenv at v0.19.0, the Load-time knob snapshot later). This is a
# correctness check, not a timing, but it holds ~40 GB, so it runs under the timing lock after the sweep.
#
#   setsid nohup bash -c 'while kill -0 "$(cat ~/goinfer-bench/release-v0.20.0/sweep-run2.pid)" 2>/dev/null; do
#     sleep 30; done; cd ~/mycode/goinfer && python3 scripts/timing_lock.py run --label release-v020-bisect-qwen35-gguf
#     -- bash ~/goinfer-bench/release-v0.20.0/run-bisect-qwen35-gguf.sh' </dev/null >~/goinfer-logs/... 2>&1 &
set -euo pipefail
GOOD=${GOOD:-c7f8eff76c7c77ee6ee8605816d1412af042cb8c} # v0.19.0
BAD=${BAD:-bcf50a49ab3d93ce92bec9460ac10f770c135aa0}   # the sweep's REV
SRC=$HOME/mycode/goinfer
BASE=$HOME/goinfer-bench/release-v0.20.0
WT=$BASE/bisect-wt
LOG=$HOME/goinfer-logs/release-v0.20.0/bisect-qwen35-gguf
SELF=$(readlink -f "$0")
PATHS=(decoder constrain internal/giw tokenizer go.mod go.sum)
export PATH=/usr/local/go/bin:$PATH GOWORK=off GOINFER_HEAVY_TESTS=1 GOINFER_NO_FIT_GUARD=1
export GOINFER_QWEN35_GGUF=$HOME/models/qwen3.6-35b-a3b-Q8_0.gguf
export GOINFER_QWEN35_GOLDEN=$HOME/models/qwen35_real_golden

# step tests the checked-out commit: 0 good, 1 bad, 125 skip (git bisect run's convention).
step() {
  local sha out line hits t0 rc=0 verdict code beat
  sha=$(git rev-parse --short=10 HEAD)
  out=$LOG/step-$sha.log
  t0=$(date +%s)
  echo "[$(date '+%H:%M:%S')] testing $sha  $(git log -1 --format=%s | cut -c1-90)" | tee -a "$LOG/progress.log"
  { echo "commit: $(git rev-parse HEAD)"; go version || true; } >"$out" 2>&1
  (while sleep 60; do echo "[$(date '+%H:%M:%S')]   ... $sha running $(( $(date +%s) - t0 ))s" >>"$LOG/progress.log"; done) &
  beat=$!
  timeout 25m go test -count=1 -tags realckpt -run '^TestQwen35GGUF_gate$' -v -timeout 20m ./decoder/ >>"$out" 2>&1 || rc=$?
  kill "$beat" 2>/dev/null || true
  line=$(grep -o 'argmax [0-9]*/80 .*worst div gap=[0-9.]*' "$out" | tail -1 || true)
  hits=$(printf '%s\n' "$line" | sed -n 's|^argmax \([0-9]*\)/80.*|\1|p')
  if [ -z "$hits" ]; then verdict=SKIP code=125
  elif [ "$hits" -ge 66 ]; then verdict=GOOD code=0
  else verdict=BAD code=1; fi
  echo "[$(date '+%H:%M:%S')]   $sha $verdict $(( $(date +%s) - t0 ))s test-rc=$rc ${line:-<no summary line, see $out>}" | tee -a "$LOG/progress.log"
  return $code
}

if [ "${1:-}" = step ]; then
  mkdir -p "$LOG"
  step
  exit $?
fi

mkdir -p "$LOG"
cd "$SRC"
git cat-file -e "$GOOD^{commit}"
git cat-file -e "$BAD^{commit}"
if [ -d "$WT" ]; then git worktree remove --force "$WT"; fi
git worktree add -q --detach "$WT" "$GOOD"
cd "$WT"
{
  echo "good:       $GOOD"
  echo "bad:        $BAD"
  echo "paths:      ${PATHS[*]}"
  echo "candidates: $(git rev-list --count "$GOOD..$BAD" -- "${PATHS[@]}")"
  echo "started:    $(date '+%F %T %Z')"
  echo "go:         $(go version)"
  echo "box:        $(hostname), $(nproc) cpus, $(free -g | awk '/Mem:/{print $2}') GB"
  echo "assets:     $GOINFER_QWEN35_GGUF, $GOINFER_QWEN35_GOLDEN"
} | tee "$LOG/provenance.txt"

if ! step; then
  echo "GOOD $GOOD did not test good on this box; nothing to bisect, stopping." | tee -a "$LOG/progress.log"
  exit 1
fi
git bisect start "$BAD" "$GOOD" -- "${PATHS[@]}"
git bisect run bash "$SELF" step 2>&1 | tee "$LOG/bisect-run.log" || true
git bisect log >"$LOG/bisect.log"
git bisect reset >/dev/null 2>&1 || true
echo "finished:   $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"

#!/usr/bin/env bash
# Night job: the GLM-OCR vision-tower cost sweep (docs/tasks/task-glm-ocr-2026-10.md, O2: "cost, measured and recorded,
# not gated"; O4 reads it to choose a default pixel cap). Pre-registered in README.md beside this script, committed
# before the first run. A RECORD, not a gate.
#
# What it times: aikit's TestGlmOcrVisionEncoder_costSweep, the wall time of one GlmOcrVisionEncoder.Forward (the
# 24-block GLM-OCR ViT, f32 reference path, quant=false, CPU) on random pixel_values at 1, 2 and 4.8 MP. The sweep runs
# 3 times, one fresh process per point, 9 processes in all, so the spread is visible. Each process waits at an
# idle gate first: the 1-min load average at most 1.0, for up to 10 min, else the job stops with NOT IDLE.
#
# aikit is PINNED at v1.52.0: the test binary is built once, from a throwaway worktree at that tag, to a durable path,
# and reused (the tree may move before tonight). night.py holds ~/.goinfer-timing.lock for the whole job; do not
# wrap this script in timing_lock.py again (a second acquire refuses).
#
# Queued with:
#   python3 scripts/night.py add glm-ocr-tower-cost --est 150 --by "Claude (Mac session), GLM-OCR O2 cost" \
#     --doc docs/tasks/task-glm-ocr-2026-10.md -- bash docs/measurements/glm-ocr-tower-cost-2026-10/run-cost.sh
# Logs: ~/goinfer-logs/glm-ocr-tower-cost-2026-10/<date>/ (provenance.txt, one raw log per process, results.txt).
set -uo pipefail
TAG=v1.52.0
AIKIT=$HOME/tmcode/aikit
CKPT=$HOME/models/glm-ocr
CKPT_REV=2e85a62840ccac27daa451df36c736c4636b8628
BASE=$HOME/goinfer-bench/glm-ocr-tower-cost-2026-10
LOG=$HOME/goinfer-logs/glm-ocr-tower-cost-2026-10/$(date +%F)
STOP_4P8_S=1800 # the first repeat's 4.8 MP point over 30 min stops the job after that point
mkdir -p "$BASE" "$LOG"

case "$CKPT" in /Volumes/*|/srv/models/*) echo "checkpoint $CKPT is on the archive, not the bench set"; exit 1;; esac
[ -z "${AIKIT_GLM_OCR:-}" ] || { echo "AIKIT_GLM_OCR is set ($AIKIT_GLM_OCR): unset it, the test must read $CKPT"; exit 1; }
for f in config.json model.safetensors; do [ -f "$CKPT/$f" ] || { echo "missing $CKPT/$f"; exit 1; }; done

HASH=$(git -C "$AIKIT" rev-parse "$TAG^{commit}") || { echo "aikit tag $TAG not found in $AIKIT"; exit 1; }
BIN=$BASE/vision-${HASH:0:12}.test
if [ ! -x "$BIN" ]; then
  WT=$BASE/wt-${HASH:0:12}
  rm -rf "$WT"; git -C "$AIKIT" worktree prune
  git -C "$AIKIT" worktree add --detach "$WT" "$HASH" || { echo "worktree at $TAG failed"; exit 1; }
  (cd "$WT" && GOWORK=off go test -c -o "$BIN" ./vision) || { echo "build of ./vision at $TAG failed"; exit 1; }
  git -C "$AIKIT" worktree remove --force "$WT"
fi

{
  echo "job:       glm-ocr-tower-cost (README.md beside run-cost.sh is the pre-registration)"
  echo "started:   $(date '+%F %T %Z')"
  echo "machine:   $(sysctl -n hw.model), $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cpus, $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
  echo "go:        $(go version)"
  echo "aikit:     $TAG = $HASH"
  echo "binary:    $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16))"
  echo "ckpt:      $CKPT (zai-org/GLM-OCR @ $CKPT_REV), local disk"
  echo "ckpt sha:  model.safetensors $(shasum -a 256 "$CKPT/model.safetensors" | cut -d' ' -f1)"
  echo "path:      f32 reference (LoadGlmOcrVisionEncoder quant=false), CPU, random pixel_values (rand seed 1)"
} | tee "$LOG/provenance.txt"

load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
gate() { # the idle gate: 1-min load <= 1.0, waiting up to 10 min
  local waited=0
  while awk -v l="$(load1)" 'BEGIN{exit !(l > 1.0)}'; do
    if [ $waited -ge 600 ]; then echo "NOT IDLE: load $(load1) after 10 min ($1)" | tee -a "$LOG/provenance.txt"; return 1; fi
    sleep 15; waited=$((waited + 15))
  done
  [ $waited -gt 0 ] && echo "gate: waited ${waited}s for load <= 1.0 ($1)" | tee -a "$LOG/provenance.txt"
  return 0
}

: > "$LOG/times.tsv"
for rep in 1 2 3; do
  for p in 1 2 4.8; do
    tag="rep$rep-${p}mp"
    gate "$tag" || exit 1
    {
      echo "== $tag $(date '+%T')"
      echo "loadavg before: $(sysctl -n vm.loadavg)"
      echo "therm before:   $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
    } | tee -a "$LOG/provenance.txt"
    AIKIT_GLM_OCR_COST=$p "$BIN" -test.run '^TestGlmOcrVisionEncoder_costSweep$' -test.v -test.count=1 -test.timeout 90m \
      > "$LOG/$tag.log" 2>&1
    rc=$?
    {
      echo "exit $rc, $(date '+%T')"
      echo "loadavg after:  $(sysctl -n vm.loadavg)"
      echo "therm after:    $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
    } | tee -a "$LOG/provenance.txt"
    wall=$(sed -n 's/.*tower wall \([0-9.]*\)s.*/\1/p' "$LOG/$tag.log" | head -1)
    if [ $rc -ne 0 ] || [ -z "$wall" ] || ! grep -q '^--- PASS' "$LOG/$tag.log"; then
      echo "FAILED $tag: exit $rc, no tower wall line or no PASS (a SKIP is not a pass); tail:" | tee -a "$LOG/provenance.txt"
      tail -5 "$LOG/$tag.log" | tee -a "$LOG/provenance.txt"
      exit 1
    fi
    printf '%s\t%s\t%s\n' "$rep" "$p" "$wall" | tee -a "$LOG/times.tsv"
    if [ $rep -eq 1 ] && [ "$p" = 4.8 ] && awk -v w="$wall" -v c=$STOP_4P8_S 'BEGIN{exit !(w > c)}'; then
      echo "STOPPED: the first 4.8 MP point took ${wall}s, over ${STOP_4P8_S}s; one repeat recorded" | tee -a "$LOG/provenance.txt"
      break 2
    fi
  done
done

python3 - "$LOG/times.tsv" <<'PY' | tee "$LOG/results.txt"
import statistics, sys
patches = {"1": 5040, "2": 10200, "4.8": 24576}
rows = [l.split("\t") for l in open(sys.argv[1]).read().split("\n") if l]
by = {}
for rep, p, w in rows:
    by.setdefault(p, []).append(float(w))
mspp = {}
for p in ("1", "2", "4.8"):
    if p not in by:
        continue
    ws = by[p]
    med = statistics.median(ws)
    mspp[p] = med * 1e3 / patches[p]
    print(f"point {p} MP ({patches[p]} patches): n={len(ws)} median {med:.2f} s, min {min(ws):.2f}, max {max(ws):.2f}; "
          f"{mspp[p]:.3f} ms/patch; all: {', '.join(f'{w:.2f}' for w in ws)}")
if "1" in mspp and "4.8" in mspp:
    r = mspp["4.8"] / mspp["1"]
    print(f"ms/patch 4.8 MP / 1 MP = {r:.2f}: " + ("the attention-quadratic term dominates (> 2)" if r > 2 else "not over 2"))
PY
echo "finished: $(date '+%F %T %Z')" | tee -a "$LOG/provenance.txt"

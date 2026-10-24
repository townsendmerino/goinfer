#!/bin/bash
# A1-G0 (docs/tasks/task-kernel-survey-2026-10.md, "KS-A1 on CUDA"): the prototype Q5_K / Q6_K GEMVs against the shipped
# gemv_w8a8_g32, built from a detached worktree at the commit named below. Run by night.py (which holds the timing lock).
set -u
REV=${1:?usage: run-a1-g0.sh <rev>}
WT=$HOME/goinfer-bench/ks-a1-run-$REV
LOG=$HOME/goinfer-logs/ks-a1-2026-10/a1-g0-$(date +%Y-%m-%d-%H%M).log
export PATH=/usr/local/go/bin:$PATH
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "$(date '+%H:%M:%S') A1-G0 at $REV, log $LOG"
cd "$HOME/mycode/goinfer" || exit 1
[ -d "$WT" ] || git worktree add -q --detach "$WT" "$REV" || exit 1
cd "$WT" || exit 1
printf 'go 1.27.0\n\ntoolchain go1.27.3\n\nuse (\n\t.\n\t./cuda\n\t./demo/agent\n\t./gpu\n\t./metal\n\t./sdktest\n)\n' > go.work
go test -c -tags 'cuda goinfer_testhooks' -o "$WT/ks-a1.test" ./cuda/ || { echo "FAILED: build"; exit 1; }
sleep 20 # let the build's load settle before the timed pass
echo "$(date '+%H:%M:%S') load $(cat /proc/loadavg); MemAvailable $(awk '/MemAvailable/{print $2/1048576 " GB"}' /proc/meminfo); $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader) used"
cd "$WT/cuda" || exit 1
GOINFER_KSA1_TIMING=1 "$WT/ks-a1.test" -test.run '^TestKSA1_gemvTiming$' -test.v -test.timeout 10m
rc=$?
echo "$(date '+%H:%M:%S') DONE rc $rc"
exit $rc

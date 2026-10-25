#!/bin/bash
# KS-H6 reading (docs/tasks/task-kernel-survey-2026-10.md, "KS-H6 reading"), through night.py (exclusive lock: no other GPU user moves
# free VRAM during it). Builds the cuda test binary at <rev> in a detached worktree and runs TestKSH6_marginParts.
set -u
REV=${1:?usage: run-h6.sh <rev> [model name filter]}
WT=$HOME/goinfer-bench/ks-h6-run-$REV
LOG=$HOME/goinfer-logs/ks-h6-2026-10/h6-$(date +%Y-%m-%d-%H%M).log
export PATH=/usr/local/go/bin:$PATH
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "$(date '+%H:%M:%S') KS-H6 at $REV, log $LOG; load $(cut -d' ' -f1-3 /proc/loadavg); MemAvailable $(awk '/MemAvailable/{printf "%.1f GB", $2/1048576}' /proc/meminfo); GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
cd "$HOME/mycode/goinfer" || exit 1
[ -d "$WT" ] || git worktree add -q --detach "$WT" "$REV" || exit 1
cd "$WT" || exit 1
printf 'go 1.27.0\n\ntoolchain go1.27.3\n\nuse (\n\t.\n\t./cuda\n\t./demo/agent\n\t./gpu\n\t./metal\n\t./sdktest\n)\n' > go.work
go test -c -tags 'cuda goinfer_testhooks' -o "$WT/cuda.test" ./cuda/ || { echo "FAILED: build"; exit 1; }
cd "$WT/cuda" && GOINFER_HEAVY_TESTS=1 GOINFER_KSH6_ONLY="${2:-}" "$WT/cuda.test" -test.run '^TestKSH6_marginParts$' -test.v -test.count 1 -test.timeout 40m
rc=$?
echo "$(date '+%H:%M:%S') DONE rc $rc"
exit $rc

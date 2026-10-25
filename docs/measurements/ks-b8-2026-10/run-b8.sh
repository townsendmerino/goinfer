#!/bin/bash
# KS-B8 build gates (docs/tasks/task-kernel-survey-2026-10.md, "KS-B8 build"), by day through night.py (which holds the timing lock).
# Builds the cuda test binary at <rev> in a detached worktree; runs B8-G1..G3 then the timed B8-G4. A skip counts as a failure.
set -u
REV=${1:?usage: run-b8.sh <rev>}
WT=$HOME/goinfer-bench/ks-b8-run-$REV
LOG=$HOME/goinfer-logs/ks-b8-2026-10/b8-$(date +%Y-%m-%d-%H%M).log
export PATH=/usr/local/go/bin:$PATH
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "$(date '+%H:%M:%S') KS-B8 gates at $REV, log $LOG"
cd "$HOME/mycode/goinfer" || exit 1
[ -d "$WT" ] || git worktree add -q --detach "$WT" "$REV" || exit 1
cd "$WT" || exit 1
printf 'go 1.27.0\n\ntoolchain go1.27.3\n\nuse (\n\t.\n\t./cuda\n\t./demo/agent\n\t./gpu\n\t./metal\n\t./sdktest\n)\n' > go.work
go test -c -tags 'cuda goinfer_testhooks' -o "$WT/cuda.test" ./cuda/ || { echo "FAILED: build"; exit 1; }
sleep 20
fail=0
gate() {
	local label=$1 re=$2; shift 2
	echo "== $label $(date '+%H:%M:%S'): load $(cut -d' ' -f1-3 /proc/loadavg); MemAvailable $(awk '/MemAvailable/{printf "%.1f GB", $2/1048576}' /proc/meminfo); GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
	(cd "$WT/cuda" && env "$@" "$WT/cuda.test" -test.run "$re" -test.v -test.count 1 -test.timeout 30m) > "$WT/$label.out" 2>&1
	local rc=$?
	cat "$WT/$label.out"
	[ $rc -eq 0 ] || { echo "FAILED: $label rc $rc"; fail=1; }
	grep -q -- '--- SKIP' "$WT/$label.out" && { echo "FAILED: $label skipped a gate"; fail=1; }
	grep -q -- '--- PASS' "$WT/$label.out" || { echo "FAILED: $label ran no test"; fail=1; }
}
gate g1 '^TestKSB8_widthChooser$'
gate g2g3 '^(TestKSB8_bitsAcrossWidths|TestKSB8_buildPricesThePass|TestPrefillChunked_bitIdentical|TestKV3_residentReportsItsChunks|TestKV3_prefillScratchBytes_matchesThePass)$' GOINFER_HEAVY_TESTS=1
gate g4 '^TestKSB8_speed$' GOINFER_HEAVY_TESTS=1 GOINFER_KSB8_TIMING=1
echo "$(date '+%H:%M:%S') DONE fail=$fail"
exit $fail

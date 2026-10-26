#!/bin/bash
# KS-B15 build gates (docs/tasks/task-kernel-survey-2026-10.md, "KS-B15 build: gates"), through night.py (which holds the timing
# lock). Builds the cuda test binary at <rev> in a detached worktree and runs B15-G1..G3, then the timed B15-G4. A skip counts as a
# failure. G3's load is the fit check: a load the card refuses at f32 KV runs at f16 (the registered fallback, ksb15Gemma3).
set -u
REV=${1:?usage: run-b15.sh <rev> [all|untimed]}
PHASE=${2:-all}
WT=$HOME/goinfer-bench/ks-b15-run-$REV
LOG=$HOME/goinfer-logs/ks-b15-2026-10/b15-$(date +%Y-%m-%d-%H%M).log
export PATH=/usr/local/go/bin:$PATH
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "$(date '+%H:%M:%S') KS-B15 gates at $REV ($PHASE), log $LOG; driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
cd "$HOME/mycode/goinfer" || exit 1
[ -d "$WT" ] || git worktree add -q --detach "$WT" "$REV" || exit 1
cd "$WT" || exit 1
printf 'go 1.27.0\n\ntoolchain go1.27.3\n\nuse (\n\t.\n\t./cuda\n\t./demo/agent\n\t./gpu\n\t./metal\n\t./sdktest\n)\n' > go.work
go vet -tags 'cuda goinfer_testhooks' ./cuda/ && go vet -tags cuda ./cuda/ || { echo "FAILED: vet"; exit 1; }
go test -c -tags 'cuda goinfer_testhooks' -o "$WT/cuda.test" ./cuda/ || { echo "FAILED: build"; exit 1; }
sleep 20
fail=0
gate() {
	local label=$1 re=$2; shift 2
	echo "== $label $(date '+%H:%M:%S'): load $(cut -d' ' -f1-3 /proc/loadavg); MemAvailable $(awk '/MemAvailable/{printf "%.1f GB", $2/1048576}' /proc/meminfo); GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
	(cd "$WT/cuda" && env "$@" "$WT/cuda.test" -test.run "$re" -test.v -test.count 1 -test.timeout 45m) > "$WT/$label.out" 2>&1
	local rc=$?
	grep -vE '^\s+kv0_invariance_test.go:[0-9]+: KV precision' "$WT/$label.out"
	[ $rc -eq 0 ] || { echo "FAILED: $label rc $rc"; fail=1; }
	grep -q -- '--- SKIP' "$WT/$label.out" && { echo "FAILED: $label skipped a gate"; fail=1; }
	grep -q -- '--- PASS' "$WT/$label.out" || { echo "FAILED: $label ran no test"; fail=1; }
	return $rc
}
gate g1 '^(TestKSB15_limitArithmetic|TestKSB15_optIn|TestAttnShmemLimit_matchesDevice|TestCheckPrefillShmem_declinesPastTheLimit|TestKV3_shmemGuardEquivalence)$'
gate g2 '^(TestKSB15_bandBitsTiny|TestKV3Tiled_batchedEqualsPerToken_tiny|TestKV3Tiled_pastBoundPromptRunsExact_tiny|TestKV3Tiled_withoutTheRowKernelsTheExactPassDeclines|TestKV3Tiled_spanScratchMatchesThePass|TestKV3Tiled_forcedRouteEqualsAttnBatched_tiny|TestKV3Families_forcedRouteEqualsAttnBatched|TestKV3Families_pastBoundBatchedEqualsPerToken|TestKV3Families_imageBlockForcedRouteEqualsAttnImgBatched|TestKV3Families_imageBlockPastTheBound|TestKV3_batchedPrefillPastTheSpanBound_tiny)$'
gate g3 '^TestKSB15_bandBitsGemma3$' GOINFER_HEAVY_TESTS=1 KSB15_REAL=1
[ "$PHASE" = all ] && [ $fail -eq 0 ] && gate g4 '^TestKSB15_speedGemma3$' GOINFER_HEAVY_TESTS=1 KSB15_TIMING=1
echo "$(date '+%H:%M:%S') DONE fail=$fail"
exit $fail

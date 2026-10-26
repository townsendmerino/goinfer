#!/bin/bash
# KS-H6 build gates (docs/tasks/task-kernel-survey-2026-10.md, "KS-H6 build: gates" and its amendment), through night.py (the timing
# lock held: memory is read, so nothing else may load the card). Builds the decoder and cuda test binaries at <rev>, and the cuda
# binary at <base> for H6C-G5's before-figures, each in a detached worktree. A skip counts as a failure, except in the readings.
set -u
REV=${1:?usage: run-h6c.sh <rev> <base>}
BASE=${2:?usage: run-h6c.sh <rev> <base>}
LOG=$HOME/goinfer-logs/ks-h6c-2026-10/h6c-$(date +%Y-%m-%d-%H%M).log
export PATH=/usr/local/go/bin:$PATH
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "$(date '+%H:%M:%S') KS-H6 gates at $REV (before-figures at $BASE), log $LOG; driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
wt() {
	local rev=$1 dir=$HOME/goinfer-bench/ks-h6c-run-$1
	(cd "$HOME/mycode/goinfer" && { [ -d "$dir" ] || git worktree add -q --detach "$dir" "$rev"; }) || return 1
	printf 'go 1.27.0\n\ntoolchain go1.27.3\n\nuse (\n\t.\n\t./cuda\n\t./demo/agent\n\t./gpu\n\t./metal\n\t./sdktest\n)\n' > "$dir/go.work"
	echo "$dir"
}
NEW=$(wt "$REV") || { echo "FAILED: worktree $REV"; exit 1; }
OLD=$(wt "$BASE") || { echo "FAILED: worktree $BASE"; exit 1; }
(cd "$NEW" && go vet -tags 'cuda goinfer_testhooks' ./cuda/ && go vet -tags cuda ./cuda/ && go vet ./decoder/) || { echo "FAILED: vet"; exit 1; }
(cd "$NEW" && go test -c -tags 'cuda goinfer_testhooks' -o "$NEW/cuda.test" ./cuda/ && go test -c -o "$NEW/decoder.test" ./decoder/) || { echo "FAILED: build"; exit 1; }
(cd "$OLD" && go test -c -tags 'cuda goinfer_testhooks' -o "$OLD/cuda.test" ./cuda/) || { echo "FAILED: build at $BASE"; exit 1; }
sleep 20
fail=0
host() { echo "load $(cut -d' ' -f1-3 /proc/loadavg); MemAvailable $(awk '/MemAvailable/{printf "%.1f GB", $2/1048576}' /proc/meminfo); GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; }
# gate <label> <binary> <pkgdir> <run> [reading] [env...]: one process; a reading may skip.
gate() {
	local label=$1 bin=$2 dir=$3 re=$4 kind=$5; shift 5
	echo "== $label $(date '+%H:%M:%S'): $(host)"
	(cd "$dir" && env "$@" "$bin" -test.run "$re" -test.v -test.count 1 -test.timeout 40m) > "$NEW/$label.out" 2>&1
	local rc=$?
	grep -vE '^\s+kv0_invariance_test.go:[0-9]+: KV precision' "$NEW/$label.out"
	[ $rc -eq 0 ] || { echo "FAILED: $label rc $rc"; fail=1; }
	if [ "$kind" = gate ]; then
		grep -q -- '--- SKIP' "$NEW/$label.out" && { echo "FAILED: $label skipped a gate"; fail=1; }
		grep -q -- '--- PASS' "$NEW/$label.out" || { echo "FAILED: $label ran no test"; fail=1; }
	fi
}
H=GOINFER_HEAVY_TESTS=1
gate s123 "$NEW/decoder.test" "$NEW/decoder" '^TestGPUProbe_' gate
gate g1 "$NEW/cuda.test" "$NEW/cuda" '^(TestKSH6C_chargedKVArithmetic|TestKSH6C_marginParts|TestFitsWeightsBudget|TestResolveCtxCapFit_shortcuts|TestKernelLocalMemoryCensus|TestKVBytesForCap|TestCheckPrefillShmem_declinesPastTheLimit)$' gate
gate tiny "$NEW/cuda.test" "$NEW/cuda" '^(TestResolveCtxCapFit_agreesWithCheckKVFits|TestCUDAKVSlots_clampedBuild|TestCUDAKVSlots_pricedAgainstWhatIsLeft|TestResolveCtxCapFit_slotsShrinkTheContext|TestCUDAKVSlots_buildTrimsTheContext|TestCUDAKVSlots_oomKeepsTheSlotsThatFit|TestWindowedKV_grantArithmetic|TestWindowedKV_grant|TestWindowedKV_expertCacheGetsTheGrant|TestCheckKVFits_realDevice_explicitRefusesWithNumbers|TestCheckKVFits_realDevice_extraResidentBytesRefusesWithNumbers|TestCheckKVFits_explicitFailsHard_defaultDeclines|TestResidentDenseBytes_matchesCUDADevice)$' gate
# H6C-S4: each model in its own process, as serve loads it and with the self-test skipped.
for spec in "qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf 16384" "qwen2.5-7b-instruct-q4_k_m.gguf 16384" "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf 0"; do
	set -- $spec
	for nost in "" 1; do
		echo "== s4 $1 ctx $2 self-test skipped '${nost}' $(date '+%H:%M:%S'): $(host)"
		(cd "$NEW/cuda" && KSH6C_NO_SELFTEST=$nost KSH6C_MODEL=$HOME/models/$1 KSH6C_CTX=$2 "$NEW/cuda.test" -test.run '^TestKSH6C_postPoolBreakdown$' -test.v -test.count 1 -test.timeout 15m 2>&1 | grep -E '^\[H6C\]|--- |FAIL|panic')
	done
done
# H6C-S5: the MoE fixture's own pool, and the plant, each in a fresh process.
gate s5 "$NEW/cuda.test" "$NEW/cuda" '^TestKSH6C_moeModelPaysItsPool$' gate KSH6C_S5=1
gate s5plant "$NEW/cuda.test" "$NEW/cuda" '^TestKSH6C_moeModelPaysItsPool$' gate KSH6C_S5=1 KSH6C_S5_PLANT=1
# H6C-G2, G3 and G5's after-figures: one process per heavy group.
gate g2small "$NEW/cuda.test" "$NEW/cuda" '^TestKSH6C_marginPriced$' gate $H KSH6C_MARGIN=1
gate g2large "$NEW/cuda.test" "$NEW/cuda" '^TestKSH6C_marginPriced7B8B$' gate $H KSH6C_MARGIN=1
gate g2moe "$NEW/cuda.test" "$NEW/cuda" '^TestKSH6C_marginPriced26B$' gate $H KSH6C_MARGIN=1
# H6C-G4: Qwen3-8B unpinned at its new context.
gate g4 "$NEW/cuda.test" "$NEW/cuda" '^TestKSH6C_headroomQwen3$' gate $H
# H6C-G5: the default unpinned contexts before and after (TestBuildScratchAccounting prints them), and Qwen3-8B pinned at 16k before.
gate g5before "$OLD/cuda.test" "$OLD/cuda" '^TestBuildScratchAccounting$' reading $H
gate g5after "$NEW/cuda.test" "$NEW/cuda" '^(TestBuildScratchAccounting|TestBuildScratchAccounting_plantedDefect)$' gate $H
gate g5before8b "$OLD/cuda.test" "$OLD/cuda" '^TestKSH6_marginPartsQwen3$' reading $H
echo "$(date '+%H:%M:%S') DONE fail=$fail"
exit $fail

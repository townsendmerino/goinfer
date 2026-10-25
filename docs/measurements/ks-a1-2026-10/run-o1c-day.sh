#!/bin/bash
# KS-O1 on CUDA (docs/tasks/task-kernel-survey-2026-10.md, "KS-O1 on CUDA: gates"), by day through night.py (which holds the
# timing lock). Builds the cuda test binary at <rev> in a detached worktree and runs one phase:
#   t0   O1C-G1 and G2 (kernels, reference, untouched bits, fused = unfused), the PTX lints, O1C-G4 (CPU unchanged: the digest at
#        <base>, then at <rev>) and O1C-G3 (memory)
#   g7   O1C-G7 (the timed A/B)
#   g56  O1C-G6 (KL toward the file) and O1C-G5 (the CUDA G-Q2), with TestQ4K_phi3ResidentMatchesCPU beside it
# Exits non-zero on any failure; a skip counts as a failure for the gates it runs.
set -u
REV=${1:?usage: run-o1c-day.sh <rev> <base> <phase>}
BASE=${2:?base}
PHASE=${3:?phase}
WT=$HOME/goinfer-bench/ks-o1c-run-$REV
LOG=$HOME/goinfer-logs/ks-a1-2026-10/o1c-$PHASE-$(date +%Y-%m-%d-%H%M).log
export PATH=/usr/local/go/bin:$PATH
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "$(date '+%H:%M:%S') O1C $PHASE at $REV (base $BASE), log $LOG"
cd "$HOME/mycode/goinfer" || exit 1
[ -d "$WT" ] || git worktree add -q --detach "$WT" "$REV" || exit 1
cd "$WT" || exit 1
printf 'go 1.27.0\n\ntoolchain go1.27.3\n\nuse (\n\t.\n\t./cuda\n\t./demo/agent\n\t./gpu\n\t./metal\n\t./sdktest\n)\n' > go.work
go test -c -tags 'cuda goinfer_testhooks' -o "$WT/cuda.test" ./cuda/ || { echo "FAILED: build"; exit 1; }
state() { echo "$(date '+%H:%M:%S') load $(cut -d' ' -f1-3 /proc/loadavg); MemAvailable $(awk '/MemAvailable/{printf "%.1f GB", $2/1048576}' /proc/meminfo); GPU $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"; }
fail=0
gate() { # gate <label> <regex> [env...]: runs the cuda binary; a SKIP of a named gate is a failure
	local label=$1 re=$2; shift 2
	state
	echo "== $label: $re"
	(cd "$WT/cuda" && env "$@" "$WT/cuda.test" -test.run "$re" -test.v -test.count 1 -test.timeout 40m) > "$WT/$label.out" 2>&1
	local rc=$?
	cat "$WT/$label.out"
	if [ $rc -ne 0 ]; then echo "FAILED: $label rc $rc"; fail=1; fi
	if [ "$label" != lints ] && grep -q -- '--- SKIP' "$WT/$label.out"; then echo "FAILED: $label skipped a gate"; fail=1; fi
	grep -q -- '--- PASS' "$WT/$label.out" || { echo "FAILED: $label ran no test"; fail=1; }
}
case "$PHASE" in
t0)
	git show "$BASE:cuda/testdata/actgroup.ptx" > "$WT/actgroup-base.ptx" || exit 1
	gate g1g2 '^(TestKSO1C_nativeKernels|TestKSO1C_referenceMatchesAikit|TestKSO1C_fusedEqualsUnfused|TestKSO1C_untouchedKernelsBitIdentical|TestActGroupKernels_.*|TestQ4K_fusedG32ResidentBitIdentical)$' GOINFER_KSO1C_OLD_ACTGROUP_PTX="$WT/actgroup-base.ptx"
	gate lints '^(TestKernelLocalMemoryCensus|TestKernelFMALint|TestKernelFMALint_coversEmbeddedPTX|TestPTX_matchesSourcesAndBindings|TestPTXModules_coverEveryEmbed)$'
	# O1C-G4: the CPU q4k digest at the base (this test file copied into a base worktree), then at REV against it.
	BW=$HOME/goinfer-bench/ks-o1c-base-$BASE
	(cd "$HOME/mycode/goinfer" && { [ -d "$BW" ] || git worktree add -q --detach "$BW" "$BASE"; }) || exit 1
	cp "$WT/go.work" "$BW/go.work"
	cp "$WT/decoder/kso1_cpu_test.go" "$BW/decoder/kso1_cpu_test.go"
	state
	echo "== g4 base"
	(cd "$BW" && GOINFER_HEAVY_TESTS=1 go test -count=1 -timeout 10m -tags goinfer_testhooks -run '^TestKSO1_cpuQ4KFingerprint$' -v ./decoder/) > "$WT/g4-base.out" 2>&1
	cat "$WT/g4-base.out"
	(cd "$BW" && git checkout -q -- decoder/kso1_cpu_test.go 2>/dev/null || rm -f decoder/kso1_cpu_test.go)
	WANT=$(sed -n 's/.*logits fingerprint \([0-9a-f]\{64\}\).*/\1/p' "$WT/g4-base.out" | head -1)
	if [ -z "$WANT" ]; then echo "FAILED: g4 base printed no digest"; fail=1; else
		echo "== g4 at $REV against the base digest $WANT"
		(cd "$WT" && GOINFER_HEAVY_TESTS=1 GOINFER_KSO1_CPU_WANT=$WANT go test -count=1 -timeout 10m -tags goinfer_testhooks -run '^TestKSO1_(cpuQ4KFingerprint|savingsCountOnlyTheLane|kquantRows|kquantServesRejects)$' -v ./decoder/) > "$WT/g4.out" 2>&1
		rc=$?; cat "$WT/g4.out"; [ $rc -eq 0 ] || { echo "FAILED: g4 rc $rc"; fail=1; }
		grep -q -- '--- SKIP' "$WT/g4.out" && { echo "FAILED: g4 skipped"; fail=1; }
	fi
	gate g3 '^TestKSO1C_laneBytes$' GOINFER_HEAVY_TESTS=1
	;;
g7)
	gate g7 '^TestKSO1C_speed$' GOINFER_KSO1C_TIMING=1
	;;
g56)
	gate g6 '^TestKSO1C_towardTheFile$' GOINFER_HEAVY_TESTS=1
	gate g5 '^(TestKSO1C_realPhi3NonInferiority|TestQ4K_phi3ResidentMatchesCPU)$' GOINFER_HEAVY_TESTS=1 GOINFER_PHI3_MINI_GGUF="$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"
	;;
*) echo "unknown phase $PHASE"; exit 2 ;;
esac
state
echo "$(date '+%H:%M:%S') DONE $PHASE fail=$fail"
exit $fail

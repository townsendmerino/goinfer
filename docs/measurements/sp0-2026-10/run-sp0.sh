#!/usr/bin/env bash
# SP0 on nobara (README.md here registers it). Builds cuda/sp0_prize_test.go at REV in its own detached worktree and runs it
# directly (not under go test, so the process's own VmHWM and peak VmSwap can be read), one target per process.
#
#   MODE=fitcheck: download and verify the Qwen3.5-9B drafter, then make every load the measurement makes, load only. Any
#                  refusal exits 75 before anything is timed.
#   MODE=measure:  the validation pairing, then Qwen3.5-9B with DFlash (template thinking, then math with thinking on), then
#                  gpt-oss-20b with n-gram, then Qwen3.6-35B-A3B with DFlash (experts streamed).
#
# The runner's checkout may predate this script, so the queued command reads it from REV:
#   python3 scripts/night.py add sp0-fitcheck --est 10 --by "mac-kapitsa, SP0" --doc docs/measurements/sp0-2026-10/README.md -- \
#     bash -c 'cd ~/goinfer-runner && git fetch -q origin && mkdir -p ~/goinfer-bench/sp0-2026-10 && git show <sha>:docs/measurements/sp0-2026-10/run-sp0.sh > ~/goinfer-bench/sp0-2026-10/run-sp0.sh && SRC=$PWD REV=<sha> MODE=fitcheck bash ~/goinfer-bench/sp0-2026-10/run-sp0.sh'
set -uo pipefail
SRC=${SRC:-$(cd "$(dirname "$0")/../../.." && pwd)} # the checkout the worktree is added from
REV=${REV:?set REV to the pinned commit}
MODE=${MODE:?set MODE to fitcheck or measure}
LOG=${LOG:-$HOME/goinfer-logs/sp0-2026-10/$MODE-$(date +%F-%H%M)}
B=${B:-$HOME/goinfer-bench/sp0-2026-10}
WT=$B/wt
export PATH=/usr/local/go/bin:$PATH
M=$HOME/models
DRAFTER9=$M/qwen35-9b-dflash
DRAFTER9_REV=5fc3b3d474760f18c516db87d84c37edbfd3ede6
DRAFTER9_SHA=0a42274b32554f48de1faa0d42824e9c2ceda649c30ae0a731cddf410dd698c7
mkdir -p "$LOG" "$B"

# Build the test binary at REV.
cd "$SRC" || exit 2
git fetch -q origin || echo "warning: git fetch failed; using the local $REV"
[ -d "$WT" ] && git worktree remove --force "$WT"
git worktree add -q --detach "$WT" "$REV" || { echo "FATAL: cannot check out $REV"; exit 2; }
printf 'go 1.27.0\n\nuse (\n\t.\n\t./gpu\n\t./cuda\n)\n' > "$WT/go.work"
(cd "$WT/cuda" && GOWORK="$WT/go.work" CGO_ENABLED=0 go test -c -tags 'cuda goinfer_testhooks' -o "$B/sp0.test" .) || { echo "FATAL: build failed"; exit 2; }
{ echo "rev:     $(git -C "$WT" rev-parse HEAD)"; echo "mode:    $MODE"; echo "started: $(date -Is)"; echo "go:      $(cd "$WT" && go version)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
  pgrep -x kwin_wayland >/dev/null && echo "desktop: kwin_wayland running (holds a CUDA context, about 16 MiB)" || echo "desktop: no compositor"; } | tee "$LOG/provenance.txt"

# The 9B drafter, pinned and checked against Hugging Face's LFS sha256.
if [ "$(sha256sum "$DRAFTER9/model.safetensors" 2>/dev/null | cut -d' ' -f1)" != "$DRAFTER9_SHA" ]; then
	echo "[$(date +%T)] downloading z-lab/Qwen3.5-9B-DFlash@$DRAFTER9_REV"
	python3 -c "import huggingface_hub as h; h.snapshot_download('z-lab/Qwen3.5-9B-DFlash', revision='$DRAFTER9_REV', local_dir='$DRAFTER9', allow_patterns=['config.json','model.safetensors','README.md'])" || { echo "FATAL: download failed"; exit 2; }
	got=$(sha256sum "$DRAFTER9/model.safetensors" | cut -d' ' -f1)
	[ "$got" = "$DRAFTER9_SHA" ] || { echo "FATAL: drafter sha256 $got, want $DRAFTER9_SHA"; exit 2; }
fi
echo "drafter: $DRAFTER9 @ $DRAFTER9_REV, sha256 $DRAFTER9_SHA" | tee -a "$LOG/provenance.txt"

rc=0
# cell <name> <env...>: the void check, then the binary with its memory sampled from /proc until it exits.
cell() {
	local name=$1; shift
	local load avail
	read -r load _ < /proc/loadavg
	avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
	echo "== $name: start $(date '+%F %T') load $load MemAvailable ${avail} MiB" | tee -a "$LOG/cells.txt"
	if awk -v l="$load" 'BEGIN{exit !(l > 8.0)}'; then
		echo "VOID: $name started at load $load (> 8.0); re-run it" | tee -a "$LOG/cells.txt"; rc=1; return
	fi
	(cd "$WT/cuda" && env GOINFER_HEAVY_TESTS=1 SP0_OUT="$LOG/$name.jsonl" "$@" \
		"$B/sp0.test" -test.run '^TestSP0_prize$' -test.v -test.count=1 -test.timeout 60m > "$LOG/$name.log" 2>&1 </dev/null) &
	local pid=$! hwm=0 swap=0
	while kill -0 $pid 2>/dev/null; do
		local child h s
		child=$(pgrep -P $pid -x sp0.test || echo $pid)
		h=$(awk '/VmHWM/ {print $2}' /proc/$child/status 2>/dev/null); s=$(awk '/VmSwap/ {print $2}' /proc/$child/status 2>/dev/null)
		[ -n "$h" ] && [ "$h" -gt "$hwm" ] && hwm=$h
		[ -n "$s" ] && [ "$s" -gt "$swap" ] && swap=$s
		sleep 2
	done
	wait $pid; local r=$?
	echo "   $name: exit $r, VmHWM $((hwm/1024)) MiB, peak VmSwap $((swap/1024)) MiB, end $(date '+%T')" | tee -a "$LOG/cells.txt"
	grep -E '^sp0 |^--- |^(PASS|FAIL)' "$LOG/$name.log" | tail -3
	if [ $r != 0 ] || ! grep -q -- '--- PASS: TestSP0_prize' "$LOG/$name.log"; then
		if [ "$MODE" = fitcheck ] && grep -q -E 'load .*: |is not CUDA-resident|AttachBlockDrafter: ' "$LOG/$name.log"; then
			echo "FIT REFUSED: $name (see $LOG/$name.log)"; tail -5 "$LOG/$name.log"; exit 75
		fi
		echo "FAILED: $name (see $LOG/$name.log)"; rc=1
	fi
}

V4=(SP0_TARGET=$M/qwen3-4b SP0_KIND=validate SP0_DRAFTER=$M/qwen3-4b-dflash-f32 SP0_SUITES=code,math)
Q9=(SP0_TARGET=$M/Qwen3.5-9B-Q4_K_M.gguf SP0_KIND=dflash SP0_DRAFTER=$DRAFTER9)
Q35=(SP0_TARGET=$M/qwen3.6-35b-a3b-q4_k_m.gguf SP0_KIND=dflash SP0_DRAFTER=$M/qwen36-35b-dflash SP0_MOE_CACHE=1 SP0_COST_REPS=10)
G20=(SP0_TARGET=$M/gpt-oss-20b-MXFP4.gguf SP0_KIND=ngram SP0_MOE_CACHE=1) # 11 GB of int4 against 8 GB: experts streamed, as TestGptOssResidentParityCUDA loads it
case $MODE in
fitcheck)
	cell fit-qwen3-4b "${V4[@]}" SP0_LOADONLY=1
	cell fit-qwen35-9b "${Q9[@]}" SP0_LOADONLY=1
	cell fit-gptoss-20b "${G20[@]}" SP0_LOADONLY=1
	cell fit-qwen36-35b "${Q35[@]}" SP0_LOADONLY=1
	;;
measure)
	cell validate-qwen3-4b "${V4[@]}"
	cell dflash-qwen35-9b "${Q9[@]}" SP0_SUITES=code,math
	cell dflash-qwen35-9b-think "${Q9[@]}" SP0_SUITES=math SP0_THINK=on SP0_SKIP_COST=1
	cell ngram-gptoss-20b "${G20[@]}" SP0_SUITES=code,math,edit
	cell dflash-qwen36-35b "${Q35[@]}" SP0_SUITES=code,math
	;;
*)
	echo "FATAL: MODE $MODE"; exit 2
	;;
esac
echo "finished: $(date -Is) rc=$rc" | tee -a "$LOG/provenance.txt"
cd "$SRC" && git worktree remove --force "$WT" 2>/dev/null
exit $rc

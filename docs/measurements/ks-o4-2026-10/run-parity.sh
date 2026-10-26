#!/usr/bin/env bash
# KS-O4's owed real-checkpoint parity on nobara's CPU (README.md here): decoder/gptoss_real_test.go's TestGptOssReal_gate and
# TestGptOssReal_logitParity (gpt-oss-20b at int8 against the HF bf16 golden from scripts/gptoss_hf_oracle.py), at the revision before
# KS-O4 (OLD) and at KS-O4 (NEW), so the record carries the cosine before and after. The golden is gitignored; it is copied (never linked)
# from the box's main checkout. Each arm is the test binary run directly, so its VmHWM and peak VmSwap can be read.
#   python3 scripts/night.py add ks-o4-parity --est 25 --by "mac-kapitsa, KS-O4" --doc docs/measurements/ks-o4-2026-10/README.md -- \
#     bash -c 'cd ~/goinfer-runner && git fetch -q origin && mkdir -p ~/goinfer-bench/ks-o4p && git show <new>:docs/measurements/ks-o4-2026-10/run-parity.sh > ~/goinfer-bench/ks-o4p/run.sh && SRC=$HOME/goinfer-runner bash ~/goinfer-bench/ks-o4p/run.sh <new> <old>'
# Exit 0 when both tests pass at NEW, 1 when one fails at NEW, 2 when it could not run. OLD's result is reported, not graded.
set -uo pipefail
NEW=${1:?the branch commit}
OLD=${2:?the commit before the change}
R=${SRC:-$HOME/goinfer-runner}
B=$HOME/goinfer-bench/ks-o4p
LOG=${LOG:-$HOME/goinfer-logs/ks-o4-2026-10/parity-$(date +%F-%H%M)}
GOLDEN=${GOLDEN:-$HOME/mycode/goinfer/decoder/testdata/gptoss_20b_golden.json}
export PATH=/usr/local/go/bin:$PATH GOWORK=off GOINFER_HEAVY_TESTS=1 GOINFER_GPTOSS_GGUF=$HOME/models/gpt-oss-20b-MXFP4.gguf
mkdir -p "$B" "$LOG"
exec > >(tee -a "$LOG/run.log") 2>&1
echo "=== $(date '+%F %T %Z') start: new $NEW, old $OLD; $(nproc) cores; $(cat /proc/loadavg)"
[ -f "$GOLDEN" ] || { echo "VOID: no golden at $GOLDEN"; exit 2; }
sha256sum "$GOLDEN"
git -C "$R" fetch -q origin || { echo "VOID: fetch failed"; exit 2; }
rc=0
for side in old new; do
	wt=$B/wt-$side rev=$NEW
	[ $side = old ] && rev=$OLD
	[ -d "$wt" ] && git -C "$R" worktree remove --force "$wt"
	git -C "$R" worktree add -q --detach "$wt" "$rev" || { echo "VOID: cannot check out $rev"; exit 2; }
	cp "$GOLDEN" "$wt/decoder/testdata/"
	(cd "$wt" && go test -c -tags realckpt -o "$B/parity-$side.test" ./decoder/) || { echo "VOID: the $side test binary did not build"; exit 2; }
	echo "== $side ($rev), $(date '+%T'): load $(cut -d' ' -f1 /proc/loadavg), available $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MiB"
	(cd "$wt/decoder" && "$B/parity-$side.test" -test.run '^TestGptOssReal_(gate|logitParity)$' -test.v -test.count=1 -test.timeout 40m \
		> "$LOG/$side.log" 2>&1 </dev/null) &
	pid=$! hwm=0 swap=0
	while kill -0 $pid 2>/dev/null; do
		child=$(pgrep -P $pid | head -1); [ -n "$child" ] || child=$pid # the subshell may exec the binary itself
		if [ -n "$child" ]; then
			h=$(awk '/VmHWM/ {print $2}' /proc/$child/status 2>/dev/null)
			s=$(awk '/VmSwap/ {print $2}' /proc/$child/status 2>/dev/null)
			[ -n "$h" ] && [ "$h" -gt "$hwm" ] && hwm=$h
			[ -n "$s" ] && [ "$s" -gt "$swap" ] && swap=$s
		fi
		sleep 1
	done
	wait $pid; r=$?
	grep -E -e '^(--- |ok|FAIL|PASS)' -e 'logit parity|loaded int8' "$LOG/$side.log"
	echo "   $side: exit $r; VmHWM $((hwm / 1024)) MiB; peak VmSwap $((swap / 1024)) MiB"
	n=$(grep -c -E -e '--- PASS: TestGptOssReal_(gate|logitParity) ' "$LOG/$side.log")
	if [ $side = new ] && { [ $r != 0 ] || [ "$n" != 2 ]; }; then
		echo "FAILED: at NEW, $n of 2 tests passed (a SKIP is not a pass)"; rc=1
	fi
	git -C "$R" worktree remove --force "$wt" 2>/dev/null
done
echo "=== $(date '+%F %T %Z') end, rc $rc; record $LOG"
exit $rc

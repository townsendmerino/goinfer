#!/usr/bin/env bash
# KS-O4's G-FID on nobara (README.md here): the pilot, N from power.py, then the gate set, through decoder/ks_o4_fidelity_test.go built at
# NEW (-tags goinfer_testhooks). The test binary runs directly so its VmHWM and peak VmSwap can be read.
#   python3 scripts/night.py add ks-o4-gfid --est 180 --not-before 22:00 --by "mac-kapitsa, KS-O4" --doc docs/measurements/ks-o4-2026-10/README.md -- \
#     bash -c 'cd ~/goinfer-runner && git fetch -q origin && mkdir -p ~/goinfer-bench/ks-o4 && git show <new>:docs/measurements/ks-o4-2026-10/run-gfid.sh > ~/goinfer-bench/ks-o4/run-gfid.sh && SRC=$HOME/goinfer-runner bash ~/goinfer-bench/ks-o4/run-gfid.sh <new>'
# N a cell is the smallest of 12, power.py's ask (at least 10) and what the 3-hour budget admits at the pilot's measured wall per prompt
# (the 2026-10-25 10:20 amendment). Exit 0 PASS, 4 AMBIGUOUS, 1 FAIL, 3 cannot resolve within the budget (10 a cell does not fit), 75 a
# load refused, 2 could not run.
set -uo pipefail
T0=$(date +%s)
BUDGET_S=${BUDGET_S:-10800} # the job's whole wall, 3 hours
NEW=${1:?the branch commit}
R=${SRC:-$HOME/goinfer-runner}
B=$HOME/goinfer-bench/ks-o4
LOG=${LOG:-$HOME/goinfer-logs/ks-o4-2026-10/gfid-$(date +%F-%H%M)}
MODEL=$HOME/models/gpt-oss-20b-MXFP4.gguf
export PATH=/usr/local/go/bin:$PATH GOWORK=off
mkdir -p "$B" "$LOG"
exec > >(tee -a "$LOG/run.log") 2>&1
echo "=== $(date '+%F %T %Z') start: $NEW; $(nproc) cores; $(cat /proc/loadavg); MemAvailable $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo) MiB"
[ -e "$MODEL" ] || { echo "VOID: $MODEL is missing"; exit 2; }
git -C "$R" fetch -q origin || { echo "VOID: fetch failed"; exit 2; }
wt=$B/wt
[ -d "$wt" ] && git -C "$R" worktree remove --force "$wt"
git -C "$R" worktree add -q --detach "$wt" "$NEW" || { echo "VOID: cannot check out $NEW"; exit 2; }
(cd "$wt" && go test -c -tags goinfer_testhooks -o "$B/kso4.test" ./decoder/) || { echo "VOID: the test binary did not build"; exit 2; }

# phase <mode> [n]: the void check at its start, then the binary with its memory sampled until it exits.
phase() {
	local mode=$1 n=${2:-12} load avail
	read -r load _ < /proc/loadavg
	avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
	echo "== $mode (N $n a cell): start $(date '+%T') load $load MemAvailable $avail MiB"
	if awk -v l="$load" 'BEGIN{exit !(l > 8.0)}'; then echo "VOID: $mode started at load $load (> 8.0)"; return 9; fi
	(cd "$wt/decoder" && env KSO4_MODEL="$MODEL" KSO4_MODE="$mode" KSO4_N="$n" KSO4_OUT="$LOG" \
		"$B/kso4.test" -test.run '^TestKSO4_gptossFidelity$' -test.v -test.count=1 -test.timeout 4h > "$LOG/$mode.log" 2>&1 </dev/null) &
	local pid=$! hwm=0 swap=0 child h s
	while kill -0 $pid 2>/dev/null; do
		child=$(pgrep -P $pid | head -1)
		if [ -n "$child" ]; then
			h=$(awk '/VmHWM/ {print $2}' /proc/$child/status 2>/dev/null); s=$(awk '/VmSwap/ {print $2}' /proc/$child/status 2>/dev/null)
			[ -n "$h" ] && [ "$h" -gt "$hwm" ] && hwm=$h
			[ -n "$s" ] && [ "$s" -gt "$swap" ] && swap=$s
		fi
		sleep 2
	done
	wait $pid; local r=$?
	echo "   $mode: exit $r, VmHWM $((hwm/1024)) MiB, peak VmSwap $((swap/1024)) MiB, end $(date '+%T')"
	grep -E '^ks-o4 ' "$LOG/$mode.log" | tail -4
	if [ $r != 0 ] || ! grep -q -- '--- PASS' "$LOG/$mode.log"; then
		grep -q -i -E 'refus|fit guard|would page' "$LOG/$mode.log" && { echo "FIT REFUSED"; exit 75; }
		echo "FAILED: $mode (see $LOG/$mode.log)"; tail -5 "$LOG/$mode.log"; exit 2
	fi
}
run() { local t=0; while :; do phase "$@"; r=$?; [ $r = 9 ] || return $r; t=$((t+1)); [ $t -ge 10 ] && { echo "VOID: stayed void"; exit 2; }; sleep 60; done; }

tp0=$(date +%s)
run pilot
tp=$(( $(date +%s) - tp0 ))
python3 "$wt/scripts/power.py" fidelity --from-positions "$LOG/pilot-positions.jsonl" | tee "$LOG/pilot-power.txt"
pooled=$(grep -o 'prompts for the gate: [0-9]*' "$LOG/pilot-power.txt" | grep -o '[0-9]*$')
[ -n "$pooled" ] || { echo "VOID: power.py gave no prompt count"; exit 2; }
ask=$(( (pooled + 1) / 2 ))
n=$ask
[ $n -lt 10 ] && n=10
[ $n -gt 12 ] && n=12
# The pilot ran one short and one long prompt twice over, with both loads: a gate prompt pair costs about half its wall.
pair=$(( tp / 2 ))
left=$(( BUDGET_S - ($(date +%s) - T0) - 600 )) # 10 min kept for the loads' variance and the report
fit=$(( pair > 0 ? left / pair : 0 ))
echo "N: power.py asks $pooled pooled ($ask a cell); the pilot took ${tp}s, so a prompt pair about ${pair}s; ${left}s of the budget left admits $fit a cell"
[ $n -gt $fit ] && n=$fit
if [ $n -lt 10 ]; then echo "CANNOT RESOLVE within the budget: $fit a cell fit in ${left}s, under the registered 10"; exit 3; fi
[ $ask -gt 12 ] && echo "note: power.py asked for $ask a cell; the gate runs at the registered cap, $n"
echo "N: $n a cell"
run gate "$n"
cat "$LOG/gate-report.txt"
v=$(grep -o '^VERDICT [A-Z]*' "$LOG/gate-report.txt" | awk '{print $2}')
git -C "$R" worktree remove --force "$wt" 2>/dev/null
echo "=== $(date '+%F %T %Z') end, verdict ${v:-none}; record $LOG"
case $v in PASS) exit 0 ;; AMBIGUOUS) exit 4 ;; *) exit 1 ;; esac

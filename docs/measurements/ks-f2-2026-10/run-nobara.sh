#!/usr/bin/env bash
# KS-F2's nobara gates (README.md here): G1b, gate identity on the CPU (amd64), and G2, the CPU decode A/B by the small-gains rule.
# Builds the branch (NEW) and the revision before it (OLD) in their own detached worktrees; the speed harness is copied from NEW into
# OLD, so both arms run the same harness. Each timed run is a process of its own, run directly so its VmHWM and peak VmSwap can be read.
#   python3 scripts/night.py add ks-f2-nobara --est 60 --by "mac-kapitsa, KS-F2" --doc docs/measurements/ks-f2-2026-10/README.md -- \
#     bash -c 'cd ~/goinfer-runner && git fetch -q origin && mkdir -p ~/goinfer-bench/ks-f2 && git show <new>:docs/measurements/ks-f2-2026-10/run-nobara.sh > ~/goinfer-bench/ks-f2/run.sh && SRC=$HOME/goinfer-runner bash ~/goinfer-bench/ks-f2/run.sh <new> <old>'
# Exit 0 when G1b and G2 ran (the verdicts are in the record), 75 when a load is refused, 1 when a gate failed or a cell is void, 2 when it
# could not run.
set -uo pipefail
NEW=${1:?the branch commit}
OLD=${2:?the commit before the change}
R=${SRC:-$HOME/goinfer-runner}
B=$HOME/goinfer-bench/ks-f2
LOG=${LOG:-$HOME/goinfer-logs/ks-f2-2026-10/nobara-$(date +%F-%H%M)}
M=$HOME/models
export PATH=/usr/local/go/bin:$PATH GOWORK=off
mkdir -p "$B" "$LOG"
exec > >(tee -a "$LOG/run.log") 2>&1
echo "=== $(date '+%F %T %Z') start: new $NEW, old $OLD; $(nproc) cores; $(cat /proc/loadavg)"
git -C "$R" fetch -q origin || { echo "VOID: fetch failed"; exit 2; }
for side in new old; do
	wt=$B/wt-$side rev=$NEW
	[ $side = old ] && rev=$OLD
	[ -d "$wt" ] && git -C "$R" worktree remove --force "$wt"
	git -C "$R" worktree add -q --detach "$wt" "$rev" || { echo "VOID: cannot check out $rev"; exit 2; }
done
cp "$B/wt-new/decoder/ksf2_speed_test.go" "$B/wt-old/decoder/"
for side in new old; do
	(cd "$B/wt-$side" && go test -c -o "$B/ksf2-$side.test" ./decoder/) || { echo "VOID: the $side test binary did not build"; exit 2; }
done
rc=0

echo "== G1b: gate identity on the CPU ($(date '+%T'))"
# The tiny fixtures a fresh worktree lacks (gitignored) are copied in (never linked) from $B/fixtures, where they were placed from the
# Mac's checkout, or from the runner's checkout. A fixture found in neither is reported; G1b's bar then counts its family as not run.
for fx in granite-tiny; do
	[ -d "$B/wt-new/testdata/$fx" ] || cp -r "$B/fixtures/$fx" "$B/wt-new/testdata/" 2>/dev/null || cp -r "$R/testdata/$fx" "$B/wt-new/testdata/" 2>/dev/null || echo "note: no $fx in $B/fixtures or $R/testdata"
done
(cd "$B/wt-new" && go run ./cmd/gate identity "$OLD" "$NEW" -backend cpu -assets tiny -steps 64 > "$LOG/g1b-identity.log" 2>&1)
g1=$?
sed -n '/== verdict ==/,$p' "$LOG/g1b-identity.log"
grep -E 'IDENTICAL|DIFFERENT|WITHIN' "$LOG/g1b-identity.log" | grep -E 'gpt-oss|granitemoehybrid|lfm2|qwen3_5_moe|nemotron_h|llama4_text'
for fam in gpt-oss granitemoehybrid lfm2 qwen3_5_moe; do
	grep -q "IDENTICAL.* $fam " "$LOG/g1b-identity.log" || { echo "G1b FAILED: $fam is not IDENTICAL"; rc=1; }
done
[ $g1 = 0 ] || { echo "G1b FAILED: gate identity exited $g1"; rc=1; }

# one <cell> <arm> <model> <depth> <tokens> <tag>: one process, the void check at its start, memory sampled until it exits.
one() {
	local cell=$1 arm=$2 model=$3 depth=$4 tokens=$5 tag=$6 load avail
	read -r load _ < /proc/loadavg
	avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
	if awk -v l="$load" 'BEGIN{exit !(l > 8.0)}'; then
		echo "VOID: $cell $tag $arm started at load $load (> 8.0)" | tee -a "$LOG/void.txt"; return 1
	fi
	local need
	need=$(cat "$LOG/need-$(basename "$model").mib" 2>/dev/null || echo 0)
	if [ "$need" -gt 0 ] && [ "$avail" -lt $((need + 2048)) ]; then
		echo "VOID: $cell $tag $arm started with ${avail} MiB available, under the fit need ${need} + 2048" | tee -a "$LOG/void.txt"; return 1
	fi
	local bin=$B/ksf2-new.test
	[ "$arm" = A ] && bin=$B/ksf2-old.test
	# GNU time reports the process's peak resident set at its exit (the 1 s /proc samples below can miss the last rise); VmSwap's peak
	# comes from the samples only.
	(cd "$B" && env KSF2_MODEL="$model" KSF2_DEPTH="$depth" KSF2_TOKENS="$tokens" KSF2_ARM="$arm" KSF2_OUT="$LOG/g2-$cell.jsonl.part" \
		/usr/bin/time -v -o "$LOG/g2-$cell-$tag-$arm.time" "$bin" -test.run '^TestKSF2_cpuDecodeSpeed$' -test.v -test.count=1 -test.timeout "$TEST_TIMEOUT" > "$LOG/g2-$cell-$tag-$arm.log" 2>&1 </dev/null) &
	local pid=$! hwm=0 swap=0 h s
	while kill -0 $pid 2>/dev/null; do
		local child
		child=$(pgrep -P $pid | head -1) # env execs the binary, so the subshell's one child is it
		if [ -n "$child" ]; then
			h=$(awk '/VmHWM/ {print $2}' /proc/$child/status 2>/dev/null)
			s=$(awk '/VmSwap/ {print $2}' /proc/$child/status 2>/dev/null)
			[ -n "$h" ] && [ "$h" -gt "$hwm" ] && hwm=$h
			[ -n "$s" ] && [ "$s" -gt "$swap" ] && swap=$s
		fi
		sleep 1
	done
	wait $pid; local r=$?
	local maxrss
	maxrss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$LOG/g2-$cell-$tag-$arm.time" 2>/dev/null)
	[ -n "$maxrss" ] && [ "$maxrss" -gt "$hwm" ] && hwm=$maxrss
	if [ "$tag" = fit ]; then echo $((hwm / 1024)) > "$LOG/need-$(basename "$model").mib"; fi
	if [ "$swap" -gt 0 ]; then
		echo "VOID: $cell $tag $arm swapped (peak VmSwap $((swap / 1024)) MiB)" | tee -a "$LOG/void.txt"; return 1
	fi
	if [ $r != 0 ] || ! grep -q -- '--- PASS' "$LOG/g2-$cell-$tag-$arm.log"; then
		echo "FAILED: $cell $tag $arm (see $LOG/g2-$cell-$tag-$arm.log)"; tail -3 "$LOG/g2-$cell-$tag-$arm.log"; return 2
	fi
	# Tag the line the harness wrote with this run's place in the order and the box's state at its start.
	python3 -c "import json,sys; l=json.loads(open(sys.argv[1]).read().splitlines()[-1]); l.update(tag=sys.argv[2], load=float(sys.argv[3]), avail_mib=int(sys.argv[4]), vmhwm_mib=int(sys.argv[5])//1024, vmswap_peak_mib=int(sys.argv[6])//1024, need_mib=int(sys.argv[7])); print(json.dumps(l))" \
		"$LOG/g2-$cell.jsonl.part" "$tag" "$load" "$avail" "$hwm" "$swap" "$need" >> "$LOG/g2-$cell.jsonl"
	echo "   $cell $tag $arm: $(tail -1 "$LOG/g2-$cell.jsonl" | python3 -c 'import json,sys; l=json.loads(sys.stdin.read()); print("%.2f ms/token, load %s, VmHWM %s MiB, swap %s MiB" % (l["decode_ms_per_token"], l["load"], l["vmhwm_mib"], l["vmswap_peak_mib"]))')"
}

# run: one, with a void run re-run in place (listed in void.txt, never averaged), at most 10 times a minute apart.
run() {
	local tries=0 r
	while :; do
		one "$@"; r=$?
		[ $r = 1 ] || return $r
		tries=$((tries + 1))
		[ $tries -ge 10 ] && { echo "VOID: $1 $6 $2 stayed void after 10 tries"; return 1; }
		sleep 60
	done
}

# The cells: depth 128 only, by Francis's call of 2026-10-25 09:58 (the depth-2048 cells cost about 8 hours each on this CPU).
cells=("gptoss-d128|$M/gpt-oss-20b-MXFP4.gguf|128")
TEST_TIMEOUT=${TEST_TIMEOUT:-20m}

echo "== fit: each cell's model at the cell's own depth, one token, its peak kept as the cell's need ($(date '+%T'))"
for c in "${cells[@]}"; do
	IFS='|' read -r cell m depth <<< "$c"
	[ -e "$LOG/need-$(basename "$m").mib" ] && continue
	run "fit-$cell" B "$m" "$depth" 1 fit || { grep -q -i -E 'refus|fit guard|would page' "$LOG"/g2-fit-*.log && { echo "FIT REFUSED: $m"; exit 75; }; echo "VOID: the fit load of $m failed"; exit 2; }
done

echo "== G2: A B B A, 2 warm runs each, then 10 pairs ($(date '+%T'))"
for c in "${cells[@]}"; do
	IFS='|' read -r cell model depth <<< "$c"
	echo "-- $cell ($(date '+%T'))"
	for w in 1 2; do run "$cell" A "$model" "$depth" 32 "warm$w" || rc=1; run "$cell" B "$model" "$depth" 32 "warm$w" || rc=1; done
	for i in $(seq 0 9); do
		if [ $((i % 2)) = 0 ]; then
			run "$cell" A "$model" "$depth" 32 "pair$i" || rc=1; run "$cell" B "$model" "$depth" 32 "pair$i" || rc=1
		else
			run "$cell" B "$model" "$depth" 32 "pair$i" || rc=1; run "$cell" A "$model" "$depth" 32 "pair$i" || rc=1
		fi
	done
done
python3 "$B/wt-new/docs/measurements/ks-f2-2026-10/grade_g2.py" "$LOG"/g2-gptoss-d128.jsonl | tee "$LOG/g2-grade.txt"
g2=${PIPESTATUS[0]}
[ "$g2" = 0 ] || { echo "G2: grade exit $g2 (FAIL, PARK or INCOMPLETE)"; rc=1; }
for side in new old; do git -C "$R" worktree remove --force "$B/wt-$side" 2>/dev/null; done
echo "=== $(date '+%F %T %Z') end, rc $rc; record $LOG"
exit $rc

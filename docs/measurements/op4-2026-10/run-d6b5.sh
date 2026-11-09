#!/bin/bash
# OP4m-D6b5 (docs/completed/task-forward-architecture-2026-10.md §4.9, "OP4m-d6 registered"): D4b-4's cell on Nemotron-H 4B
# Instruct, an int4 Metal .giw from ~/models, through TestOP4mD6b_mambaStepSpeed, under the void rule (scripts/void_rule.py;
# docs/measurements/noise-registry.md §8). Run it inside the exclusive timing lock:
#
#   BIN=<metal test binary built at the cell's commit> MODEL=$HOME/models/nemotron-h-4b-instruct-128k.int4.metal.v17.giw \
#   OUT=<durable dir> python3 scripts/timing_lock.py run --wait 7200 --label mac-fit-op4m-d6b5 -- \
#     bash docs/measurements/op4-2026-10/run-d6b5.sh
#
# Each attempt: wait for the 1-minute load to come under the cap (not an attempt), start the swap watcher, `void_rule.py
# start` (load and available memory against NEED_MB), the cell, stop the watcher, `void_rule.py end` (swap growth over the
# cell, sampled). A void attempt (refused at the start, the cell's own "cell void", or swap growth) is listed and re-run, up to
# six attempts (the registration's stopping rule). Exit 0: the cell counted and passed; 1: it counted and did not pass, or
# an error that is not a void; 75: six voids.
set -u
: "${BIN:?BIN: the metal test binary}" "${MODEL:?MODEL: the .giw under ~/models}" "${OUT:?OUT: a durable directory}"
case "$MODEL" in
/Volumes/* | /srv/models/*)
	echo "refused: $MODEL is on the archive, not the bench set"
	exit 2
	;;
esac
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
VR="python3 $REPO/scripts/void_rule.py"
# The need: the fit check's Metal plan at ctx 1024, int4, one slot (dense 3.40 GB, recurrent state 0.08 GB, scratch 10.69 MB,
# host copy 1.63 GB, KV 0.02 GB: 5.14 GB), plus three further slots' recurrent state and KV (0.10 GB each), rounded up. The
# cell checks its own exact need (ResidentNeedBytes with every slot, then the step's scratch) by the same rule.
NEED_MB=5600
NEED_SRC="goinfer-chat fit, metal, ctx 1024, int4, plus three slots"
LOAD_CAP=4.0
mkdir -p "$OUT"
echo "run-d6b5: BIN $BIN ($(shasum -a 256 "$BIN" | cut -c1-16)), MODEL $MODEL, OUT $OUT, need $NEED_MB MB, load cap $LOAD_CAP, $(date '+%F %T %Z')"
for attempt in 1 2 3 4 5 6; do
	A="$OUT/attempt-$attempt"
	for _ in $(seq 1 80); do # the idle gate: up to 20 min, 15 s a poll
		l=$(sysctl -n vm.loadavg | awk '{print $2}')
		awk -v l="$l" -v c="$LOAD_CAP" 'BEGIN{exit !(l<=c)}' && break
		sleep 15
	done
	$VR watch --json-out "$A.peak.json" --ready-file "$A.ready" &
	W=$!
	if ! $VR wait --ready-file "$A.ready"; then
		echo "attempt $attempt: VOID: the swap watcher never became ready"
		kill -TERM $W 2>/dev/null
		wait $W
		continue
	fi
	if ! $VR start --need-mb $NEED_MB --need-source "$NEED_SRC" --load-cap $LOAD_CAP --json-out "$A.start.json" 2>"$A.start.err"; then
		echo "attempt $attempt: VOID at the start ($(date '+%T')): $(cat "$A.start.err")"
		kill -TERM $W 2>/dev/null
		wait $W
		continue
	fi
	echo "attempt $attempt: cell start $(date '+%T'): $(cat "$A.start.json")"
	cd "$REPO/backend/metal" || exit 1 # the package directory, as go test runs it
	GOINFER_OP4MD6B_TIMING=1 GOINFER_OP4MD6B_MODEL="$MODEL" "$BIN" -test.run '^TestOP4mD6b_mambaStepSpeed$' -test.v -test.count=1 \
		-test.timeout 20m >"$A.log" 2>&1
	trc=$?
	kill -TERM $W 2>/dev/null
	wait $W
	$VR end --start-json "$A.start.json" --peak-json "$A.peak.json" --json-out "$A.end.json" 2>"$A.end.err"
	erc=$?
	echo "attempt $attempt: cell end $(date '+%T'): test exit $trc, end exit $erc: $(cat "$A.end.json" 2>/dev/null) $(cat "$A.end.err")"
	sed -n '/\[OP4m-d6b D6b-5\]/p; /^--- /p; /cell void/p' "$A.log"
	if [ $erc -ne 0 ]; then
		echo "attempt $attempt: VOID at the end (swap growth or an unreadable reading); re-run"
		continue
	fi
	if grep -q "cell void" "$A.log"; then
		echo "attempt $attempt: VOID in the cell; re-run"
		continue
	fi
	if [ $trc -eq 0 ] && grep -q -- "--- PASS: TestOP4mD6b_mambaStepSpeed" "$A.log"; then
		echo "run-d6b5: COUNTED attempt $attempt: PASS"
		exit 0
	fi
	if grep -q -E "PARKED|FAIL \(median under 1.00\)" "$A.log"; then
		echo "run-d6b5: COUNTED attempt $attempt: not a pass (see the verdict line)"
		exit 1
	fi
	echo "run-d6b5: attempt $attempt ended in an error that is not a void (test exit $trc); stopping"
	tail -20 "$A.log"
	exit 1
done
echo "run-d6b5: six attempts void; the cell moves to night.py once (the registration)"
exit 75

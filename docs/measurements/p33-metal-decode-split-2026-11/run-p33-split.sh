#!/usr/bin/env bash
# P33's first measurement (registration.md beside this file): where a Metal decode token's time goes on Qwen2.5-Coder-1.5B at
# int4. Three cells from one test binary built at the run's commit with -tags goinfer_testhooks, each under the void rule
# (scripts/void_rule.py; docs/measurements/noise-registry.md §8), the whole run inside the exclusive timing lock:
#
#   (cd backend/metal && go test -c -tags goinfer_testhooks -o <dir>/metal-tagged-<rev>.test .)
#   BIN=<dir>/metal-tagged-<rev>.test OUT=<durable dir> NEED_MB=<the fit check's plan for the load, MB> \
#     python3 scripts/timing_lock.py run --wait 1800 --label p33-split -- \
#     bash docs/measurements/p33-metal-decode-split-2026-11/run-p33-split.sh
#
# Cells, in the registered order:
#   gemv-s0   TestMetalGEMVS0: the streaming-read ceiling and each production GEMV kernel alone at its real shape (no checkpoint)
#   decomp    TestMetalDecodeDecomp at depths 128 and 192: each category's in-sequence GPU work, the dispatch floor, the full token
#   chain-ab  TestCB01ChainAB, greedy: the decode rate through Generate, the greedy chain against the full-logits path
# A cell refused at its start, or one during which system swap grew, is void: it is listed and run again once (the registration's
# stopping rule). A test that skips is not a pass. Exit 0: all three cells counted; 75: a cell stayed void; 1: a test did not pass.
set -u
: "${BIN:?BIN: the metal test binary}" "${OUT:?OUT: a durable directory}" "${NEED_MB:?NEED_MB: the fit plan for the load, in MB}"
MODEL=${MODEL:-$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v18.giw}
case "$MODEL" in
/Volumes/* | /srv/models/*)
	echo "refused: $MODEL is on the archive, not the bench set"
	exit 2
	;;
esac
[ -x "$BIN" ] || { echo "missing test binary $BIN"; exit 2; }
[ -f "$MODEL" ] || { echo "missing model $MODEL"; exit 2; }
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
VR="python3 $REPO/scripts/void_rule.py"
LOAD_CAP=4.0
GEMV_NEED_MB=1024 # TestMetalGEMVS0 allocates up to about 1 GB of device buffers and loads no model
mkdir -p "$OUT"
{
	echo "tree:     $(git -C "$REPO" rev-parse HEAD) ($(git -C "$REPO" status --porcelain | wc -l | tr -d ' ') git status entries)"
	echo "binary:   $BIN (sha256 $(shasum -a 256 "$BIN" | cut -c1-16); $(go version "$BIN" 2>/dev/null | awk '{print $2}'))"
	echo "model:    $MODEL ($(stat -f %z "$MODEL") bytes)"
	echo "need:     $NEED_MB MB for the two cells that load it; $GEMV_NEED_MB MB for gemv-s0"
	echo "started:  $(date '+%F %T %Z')"
	echo "machine:  $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
	echo "load:     $(sysctl -n vm.loadavg)"
	echo "swap:     $(sysctl -n vm.swapusage)"
	echo "therm:    $(pmset -g therm 2>/dev/null | tr '\n' ' ')"
} | tee "$OUT/provenance.txt"
note() { echo "$*" | tee -a "$OUT/provenance.txt"; }

counted=0
voided=0
failed=0
cell() { # <name> <need MB> <need source> <test name> [ENV=value ...]
	local name=$1 need=$2 src=$3 test=$4
	shift 4
	local attempt A W l trc erc
	for attempt in 1 2; do
		A="$OUT/$name.attempt-$attempt"
		for _ in $(seq 1 40); do # the idle gate: up to 10 min, 15 s a poll; a wait here is not an attempt
			l=$(sysctl -n vm.loadavg | awk '{print $2}')
			awk -v l="$l" -v c="$LOAD_CAP" 'BEGIN{exit !(l<=c)}' && break
			sleep 15
		done
		$VR watch --json-out "$A.peak.json" --ready-file "$A.ready" &
		W=$!
		if ! $VR wait --ready-file "$A.ready"; then
			note "== $name attempt $attempt: VOID: the swap watcher never became ready"
			kill -TERM $W 2>/dev/null
			wait $W
			continue
		fi
		if ! $VR start --need-mb "$need" --need-source "$src" --load-cap $LOAD_CAP --json-out "$A.start.json" 2>"$A.start.err"; then
			note "== $name attempt $attempt: VOID at the start ($(date '+%T')): $(cat "$A.start.err")"
			kill -TERM $W 2>/dev/null
			wait $W
			continue
		fi
		note "== $name attempt $attempt: start $(date '+%T'): $(cat "$A.start.json")"
		(cd "$REPO/backend/metal" && env "$@" "$BIN" -test.run "^$test\$" -test.v -test.count=1 -test.timeout 20m) >"$A.log" 2>&1
		trc=$?
		kill -TERM $W 2>/dev/null
		wait $W
		$VR end --start-json "$A.start.json" --peak-json "$A.peak.json" --json-out "$A.end.json" 2>"$A.end.err"
		erc=$?
		note "== $name attempt $attempt: end $(date '+%T'): test exit $trc, end exit $erc: $(cat "$A.end.json" 2>/dev/null) $(cat "$A.end.err")"
		if [ $erc -ne 0 ]; then
			note "== $name attempt $attempt: VOID at the end (swap growth or an unreadable reading)"
			continue
		fi
		if [ $trc -eq 0 ] && grep -q -- "^--- PASS: $test " "$A.log"; then
			cp "$A.log" "$OUT/$name.log"
			counted=$((counted + 1))
			note "== $name: COUNTED attempt $attempt"
			return 0
		fi
		note "== $name attempt $attempt: the test did not pass (exit $trc; a skip is not a pass)"
		tail -20 "$A.log"
		failed=$((failed + 1))
		return 1
	done
	voided=$((voided + 1))
	note "== $name: two attempts void; the attempt ends for the day (the registration's stopping rule)"
	return 75
}

cell gemv-s0 "$GEMV_NEED_MB" "the test's own device buffers, about 1 GB" TestMetalGEMVS0 GOINFER_METAL_GEMV_S0=1
cell decomp "$NEED_MB" "the fit check's plan" TestMetalDecodeDecomp GOINFER_METAL_DDECOMP=1 "GOINFER_METAL_DDECOMP_MODEL=$MODEL" \
	GOINFER_METAL_DDECOMP_DEPTHS=128,192
cell chain-ab "$NEED_MB" "the fit check's plan" TestCB01ChainAB GOINFER_METAL_CB01_AB=1 "GOINFER_METAL_CB01_MODEL=$MODEL"

{
	for f in gemv-s0 decomp chain-ab; do
		[ -f "$OUT/$f.log" ] || continue
		echo "##### $f"
		grep -E 'streaming read|^=== |^  |METRIC|identity:|loaded |^--- ' "$OUT/$f.log"
	done
} >"$OUT/results.txt"
cat "$OUT/results.txt"
note "finished: $(date '+%F %T %Z'); $counted of 3 cells counted, $voided void, $failed not passed"
[ "$counted" -eq 3 ] && exit 0
[ "$failed" -gt 0 ] && exit 1
exit 75

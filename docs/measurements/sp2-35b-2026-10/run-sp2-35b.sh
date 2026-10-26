#!/usr/bin/env bash
# SP2 on the 35B (docs/measurements/sp2-35b-2026-10/README.md): the SP2 harness with the 35B, its drafter, w = 3 and experts streamed.
#   MODE=identity  G1 (SP2's G1c + plants + G5) on the 35B, with G3's load record (correctness; the timing lock exclusive: it needs the card to itself)
#   MODE=speed     G4, speed against plain decode at w = 3 (timed; exclusive lock)
# The test binary runs directly (go test -c), and its VmHWM and peak VmSwap are sampled until it exits.
set -u
MODE=${MODE:?MODE=identity|speed}
REPO=${REPO:-/home/francis/goinfer-bench/sp2-35b-2026-10}
OUT=${OUT:-$HOME/goinfer-logs/sp2-35b-2026-10}
export PATH=/usr/local/go/bin:$PATH GOPROXY=off GOINFER_HEAVY_TESTS=1
export SP2_TARGET=$HOME/models/qwen3.6-35b-a3b-q4_k_m.int4.e4h.cuda.v17.giw SP2_DRAFTER=$HOME/models/qwen36-35b-dflash SP2_WIDTH=3 SP2_MOE_STREAM=1
mkdir -p "$OUT"
stamp=$(date +%Y-%m-%d-%H%M)
case $MODE in
identity) run='^TestSP2_identity$'; lock=; tmo=60m ;;
speed) run='^TestSP2_speed$'; lock=; tmo=150m ;;
*) echo "MODE $MODE?"; exit 2 ;;
esac
cd "$REPO/cuda" || exit 1
bin="$OUT/sp2-35b-cuda-$(git rev-parse --short HEAD).test"
[ -x "$bin" ] || python3 "$REPO/scripts/timing_lock.py" run --shared --wait 14400 --label sp2-35b-build -- \
	go test -c -tags 'cuda goinfer_testhooks' -o "$bin" . || exit 1
export SP2_OUT="$OUT/$MODE-$stamp.jsonl"
log="$OUT/$MODE-$stamp.log"
{
	echo "# $MODE at $(git rev-parse HEAD), $(date -Is)"
	echo "# load $(cut -d' ' -f1-3 /proc/loadavg); MemAvailable $(awk '/MemAvailable/{print $2" kB"}' /proc/meminfo); driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
	nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/# gpu: /'
} > "$log"
python3 "$REPO/scripts/timing_lock.py" run $lock --wait 14400 --label "sp2-35b-$MODE" -- \
	bash -c '"$0" -test.run "$1" -test.v -test.timeout "$2" >> "$3" 2>&1 & pid=$!; hwm=0; sw=0
	while kill -0 $pid 2>/dev/null; do
		h=$(awk "/VmHWM/{print \$2}" /proc/$pid/status 2>/dev/null); s=$(awk "/VmSwap/{print \$2}" /proc/$pid/status 2>/dev/null)
		[ -n "$h" ] && [ "$h" -gt "$hwm" ] && hwm=$h; [ -n "$s" ] && [ "$s" -gt "$sw" ] && sw=$s; sleep 2
	done; wait $pid; rc=$?; echo "# exit $rc; VmHWM ${hwm} kB; peak VmSwap ${sw} kB; $(date -Is)" >> "$3"; exit $rc' \
	"$bin" "$run" "$tmo" "$log"

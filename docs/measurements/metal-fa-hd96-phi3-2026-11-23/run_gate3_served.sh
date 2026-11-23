#!/bin/bash
# Gate 3 for Metal's hd-96 attention_fa (README beside this file), by day, under the Mac's exclusive timing lock:
#   python3 scripts/timing_lock.py run --label hd96-metal-gate3 -- bash docs/measurements/metal-fa-hd96-phi3-2026-11-23/run_gate3_served.sh
# CUDA's run_gate3_served.sh (fa-hd96-phi3-2026-11-22) on Metal: one serve binary for both arms, goinfer with -flash-decode-hd96 and
# goinfer_old with =false, Phi-3 mini at its default, depths 128 and 3,900, two runs a cell, one quad. Grades with CUDA's grade_cell.py.
set -uo pipefail
BIN=${BIN:?set BIN to the serve binary}
OUT=${OUT:-$HOME/goinfer-bench/hd96/served-$(date +%H%M)}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}
mkdir -p "$OUT/serve-logs" || exit 2
[ -x "$BIN" ] || { echo "FATAL: $BIN is missing"; exit 2; }
M=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
[ -f "$M" ] || { echo "FATAL: the checkpoint is missing"; exit 2; }
case "$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$M")" in /Volumes/*|/srv/models*) echo "FATAL: archive path"; exit 2;; esac
load1() { sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1}'; }
settle() { for _ in $(seq 1 60); do awk -v l="$(load1)" 'BEGIN{exit !(l<2.0)}' && return 0; sleep 10; done; return 1; }
{ echo "started: $(date '+%F %T %Z')"; echo "binary: $BIN sha256 $(shasum -a 256 "$BIN" | cut -c1-16)"
  echo "host: $(hostname) macOS $(sw_vers -productVersion)"; echo "load: $(sysctl -n vm.loadavg)"; echo "power: $(pmset -g batt | head -1)"
  echo "swap: $(sysctl -n vm.swapusage)"; } | tee "$OUT/provenance.txt"
settle || { echo "!! the box did not settle under load 2.0: $(load1)"; exit 2; }
python3 "$HERE/probe_lane_metal.py" "$BIN" "$OUT" 2>&1 | tee "$OUT/probe.log"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "!! the arms did not dispatch their own attention kernels: the cell does not run"; exit 2; }
settle || { echo "!! the box did not settle after the probe: $(load1)"; exit 2; }
rc=0
(cd "$REPO" && env BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_MODELS=phi3-mini BENCH_DEPTHS=3900 \
  BENCH_QUANT_OVERRIDE="phi3-mini=default" BENCH_RUNS=2 BENCH_ABBA=1 BENCH_SERVE_LOG_DIR="$OUT/serve-logs" BENCH_LOAD_TIMEOUT=240 \
  GOINFER_SERVE_METAL="$BIN" GOINFER_SERVE_METAL_OLD="$BIN" BENCH_GOINFER_ARGS="-flash-decode-hd96" BENCH_GOINFER_OLD_ARGS="-flash-decode-hd96=false" \
  python3 -u scripts/bench_peer.py "$OUT/decode-metal-all.json") 2>&1 | tee "$OUT/decode-metal.log"
[ "${PIPESTATUS[0]}" = 0 ] && [ -s "$OUT/decode-metal-all.json" ] || { echo "!! the run failed or wrote no results (bench_peer exits 0 when it refuses a busy box)"; rc=1; }
# CUDA's grader reads decode-cuda-*.json; the Metal file under that name, the same rows.
cp "$OUT/decode-metal-all.json" "$OUT/decode-cuda-all.json"
python3 "$HERE/../fa-hd96-phi3-2026-11-22/grade_cell.py" "$OUT" 2>&1 | tee "$OUT/verdict.txt"
[ "${PIPESTATUS[0]}" = 0 ] || rc=1
echo "finished: $(date '+%F %T %Z') rc=$rc" | tee -a "$OUT/provenance.txt"
exit $rc

#!/usr/bin/env bash
# Run repro.py against one goinfer-serve binary with gpt-oss-20b on nobara's card (README.md, this directory).
#   python3 scripts/timing_lock.py run --label <name> -- bash run.sh <label> <cases>
# Env: SERVE (the binary), SRC (a checkout at 5df3b780 for BIG1's prompts), OUT (output root), PORT.
set -u
label=$1 cases=$2
here=$(cd "$(dirname "$0")" && pwd)
SERVE=${SERVE:?set SERVE} SRC=${SRC:?set SRC}
OUT=${OUT:-$HOME/goinfer-logs/mm01/gptoss-forced-call} PORT=${PORT:-18293}
model=$HOME/models/gpt-oss-20b-MXFP4.gguf
dir=$OUT/$label-$(date +%H%M); mkdir -p "$dir"
log() { echo "$(date '+%F %T') $*" | tee -a "$dir/run.log"; }
log "start: $label, $(basename "$SERVE") $(sha256sum "$SERVE" | cut -c1-12), cases $cases"
load1=$(cut -d' ' -f1 /proc/loadavg); avail=$(awk '/^MemAvailable:/{printf "%.1f", $2*1024/1e9}' /proc/meminfo)
apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E "kwin_wayland|Xwayland|plasmashell")
log "load $load1; MemAvailable $avail GB; driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader); card: ${apps:-only the compositor}"
[[ -n "$apps" ]] && { log "VOID at start: card busy: $apps"; exit 6; }
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 8.0 else 1)" "$load1" && { log "VOID at start: load $load1 > 8.0"; exit 6; }
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 15 else 1)" "$avail" && { log "VOID at start: MemAvailable $avail GB < 13 GB need + 2"; exit 6; }
"$SERVE" -addr 127.0.0.1:$PORT -model "gptoss=$model" -backend cuda -require-backend -moe-cache-experts -ctx 32768 >"$dir/serve.log" 2>&1 </dev/null &
spid=$!
python3 "$SRC/docs/measurements/big01-moe-serve-2026-10-30/memwatch.py" $spid "$dir/mem.log" &
mpid=$!
t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"gptoss"'; do
  kill -0 $spid 2>/dev/null || { log "server exited before it served"; wait $mpid; exit 3; }
  [ $(( $(date +%s) - t0 )) -gt 600 ] && { log "no model after 600 s"; kill $spid; wait $mpid; exit 4; }
  sleep 2
done
log "serving after $(( $(date +%s) - t0 )) s"
grep -E 'fit:|KV plan|features:|swap guard' "$dir/serve.log" | tee -a "$dir/run.log"
timeout 1800 python3 "$here/repro.py" "http://127.0.0.1:$PORT" gptoss "$SRC" "$dir/repro.jsonl" "$cases" 2>&1 | tee -a "$dir/run.log"
prc=${PIPESTATUS[0]}
kill $spid 2>/dev/null; wait $spid; wait $mpid
grep -E '^(peak|VOID)' "$dir/mem.log" | tee -a "$dir/run.log"
log "done (repro exit $prc); record in $dir"
exit $prc

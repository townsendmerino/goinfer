#!/bin/bash
# BIG1 serve check (README.md, this directory): load one model on nobara's card with goinfer-serve, watch the server's
# memory and swap the whole time (memwatch.py stops it on swap growth), run probe.py against it, stop it.
# Run it under the timing lock, which it holds for the whole load:
#   python3 scripts/timing_lock.py run --label big01-<name> -- bash run-serve-check.sh <name> <model-path> [serve args...]
# Env: SERVE (the goinfer-serve binary), OUT (the output root), PORT.
set -u
name=$1 model=$2; shift 2
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
SERVE=${SERVE:?set SERVE to the goinfer-serve binary}
OUT=${OUT:-$HOME/goinfer-bench/big01-2026-10-30}
PORT=${PORT:-18291}
dir=$OUT/$name-$(date +%H%M)
mkdir -p "$dir"
log() { echo "$(date '+%F %T') $*" | tee -a "$dir/run.log"; }
log "start: $name, $model, $(basename "$SERVE") $(sha256sum "$SERVE" | cut -c1-12), args: $*"
log "load $(cut -d' ' -f1-3 /proc/loadavg); $(free -g | awk '/^Mem:/{print $7" GB available"}'); swap $(free -m | awk '/^Swap:/{print $3" MB used"}'); VRAM used, total: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
"$SERVE" -addr 127.0.0.1:$PORT -model "$name=$model" -backend cuda -require-backend "$@" >"$dir/serve.log" 2>&1 </dev/null &
spid=$!
python3 "$here/memwatch.py" $spid "$dir/mem.log" &
mpid=$!
t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "\"$name\""; do
  if ! kill -0 $spid 2>/dev/null; then log "server exited before it served (see serve.log)"; wait $mpid; exit 3; fi
  if [ $(( $(date +%s) - t0 )) -gt 900 ]; then log "no model after 900 s; stopping"; kill $spid; wait $mpid; exit 4; fi
  sleep 2
done
log "serving after $(( $(date +%s) - t0 )) s"
grep -E 'fit:|C′ cache|decode path|loaded "|swap guard|KV plan|context:' "$dir/serve.log" | tee -a "$dir/run.log"
timeout 2400 python3 "$here/probe.py" "http://127.0.0.1:$PORT" "$name" "$repo" "$dir/probe.jsonl" 2>&1 | tee -a "$dir/run.log"
prc=${PIPESTATUS[0]}
log "probe exit $prc"
grep -E 'swap guard' "$dir/serve.log" | tee -a "$dir/run.log"
kill $spid 2>/dev/null
wait $spid
wait $mpid
grep -E '^(peak|VOID)' "$dir/mem.log" | tee -a "$dir/run.log"
log "done; record in $dir"
if grep -q '^VOID' "$dir/mem.log"; then exit 5; fi
exit $prc

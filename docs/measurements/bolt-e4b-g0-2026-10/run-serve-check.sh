#!/usr/bin/env bash
# `serve check` of Nemotron 3.5 Bolt E4B Q4_K_M on nobara's CUDA card: start the pinned server (its banner is the fit check), wait
# for it, run the check (harness rows included), stop the server by pid. Run under the timing lock:
#   python3 scripts/timing_lock.py run --wait 7200 --label bolt-serve-check -- bash docs/measurements/bolt-e4b-g0-2026-10/run-serve-check.sh
set -uo pipefail
BIN=${BIN:-$HOME/goinfer-bench/bolt-g0-run/serve-cuda}
MODEL=${MODEL:-$HOME/models/nemotron-3.5-bolt-e4b-gguf/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf}
LOG=${LOG:-$HOME/goinfer-logs/bolt-e4b-g0-2026-10-24}
PORT=${PORT:-18321}
mkdir -p "$LOG"
snap() { # $1 = label: load average, free memory and VRAM at this point
	{ echo "== $1 $(date -Is)"; cat /proc/loadavg; free -m | sed -n 2p
	  nvidia-smi --query-gpu=driver_version,memory.used,memory.total --format=csv,noheader; } >> "$LOG/context.txt"
}
{ echo "binary rev: $(cat "$(dirname "$BIN")/rev" 2>/dev/null)"; echo "model: $MODEL"; sha256sum "$MODEL"; } > "$LOG/context.txt"
snap start
echo "[$(date +%T)] starting server"
"$BIN" -model "$MODEL" -addr 127.0.0.1:$PORT -ctx 16384 > "$LOG/serve.log" 2>&1 &
SRV=$!
up=0
for _ in $(seq 1 900); do
	if grep -q 'goinfer serving on' "$LOG/serve.log"; then up=1; break; fi
	kill -0 $SRV 2>/dev/null || break
	sleep 1
done
if [ $up != 1 ]; then
	echo "[$(date +%T)] FAILED: the server did not come up (refused or crashed); tail of serve.log:"; tail -20 "$LOG/serve.log"
	kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; exit 1
fi
echo "[$(date +%T)] server up"; sed -n '1,20p' "$LOG/serve.log"
snap server-up
"$BIN" check -timeout 15m http://127.0.0.1:$PORT 2>&1 | tee "$LOG/check.txt"
RC=${PIPESTATUS[0]}
snap after-check
kill $SRV; wait $SRV 2>/dev/null
echo "[$(date +%T)] check exit=$RC"
exit $RC

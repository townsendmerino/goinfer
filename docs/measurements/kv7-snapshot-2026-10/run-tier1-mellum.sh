#!/bin/bash
# Tier 1 of the prefix snapshot, Mellum2.1 on the 8 GB card (docs/measurements/kv7-snapshot-2026-10/kv7-tier1-2026-10-16.md): the first
# requests of two opencode and two Codex sessions, replayed in the order oc, cx, oc, cx against one server, with snapshots off
# (-kv-host-bytes 0) and on (auto). Each arm starts its own server from the same pinned binary and flags as the G-ids-3 and fast-alias runs.
#
#   B=$HOME/goinfer-bench/kv7-2026-10 bash run-tier1-mellum.sh        (hold the timing lock around it: timing_lock.py run)
set -u
B=${B:?the directory holding bin/serve-cuda-kv7}
BIN=$B/bin/serve-cuda-kv7
OUT=${OUT:-$B/out-mellum}
MODEL=${MODEL:-$HOME/models/mellum2.1-thinking}
PORT=${PORT:-18090}
OC=$HOME/goinfer-bench/fast-alias-2026-10/out/bodies-cc-opencode-mellum-fast
CX=$HOME/goinfer-bench/ids3-2026-10/ids3-out
HERE=$(cd "$(dirname "$0")" && pwd)
for f in "$BIN" "$MODEL/config.json" "$OC/00002.req.json.gz" "$OC/00006.req.json.gz" "$CX/bodies-ids3-on-read-r1/00001.req.json.gz" "$CX/bodies-ids3-on-search-r1/00001.req.json.gz" "$HERE/replay.py"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }
done
[ -z "$(ls "$HOME"/models | grep -E '\.v16\.giw$' )" ] || { echo "FATAL: a .v16.giw sidecar is under ~/models; this build writes v17"; exit 2; }
mkdir -p "$OUT" || exit 2
echo "binary $(cat "$BIN.rev") sha256 $(cat "$BIN.sha256")" > "$OUT/provenance.txt"
{ date '+start %F %T %Z'; uname -srm; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader; free -m | sed -n 1,3p; echo "gpu apps: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"; } >> "$OUT/provenance.txt"
for ARM in off on; do
  D=$OUT/$ARM; mkdir -p "$D"
  HB=auto; [ "$ARM" = off ] && HB=0
  echo "[$(date +%T)] arm $ARM: starting the server (-kv-host-bytes $HB)"
  "$BIN" --model "$MODEL" --backend cuda --ctx 16384 --moe-cache-experts --addr 127.0.0.1:$PORT --api-key goinfer --allow-admin \
    --log-requests --log-token-ids "$D/token-ids.jsonl" --kv-host-bytes "$HB" > "$D/serve.log" 2>&1 < /dev/null &
  PID=$!; echo $PID > "$D/serve.pid"
  for i in $(seq 1 240); do
    curl -s -m 2 -H 'authorization: Bearer goinfer' http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 && break
    kill -0 $PID 2>/dev/null || { echo "FATAL: the server exited during startup"; tail -5 "$D/serve.log"; exit 1; }
    sleep 2
  done
  curl -s -m 2 -H 'authorization: Bearer goinfer' http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 || { echo "FATAL: no server after 480 s"; kill $PID; exit 1; }
  grep -E "prefix snapshots|fit:|KV plan|windowed KV" "$D/serve.log" | cut -c1-200
  python3 -I "$HERE/replay.py" --base http://127.0.0.1:$PORT --pid $PID --out "$D" --max-tokens 24 \
    oc-read=chat="$OC/00002.req.json.gz" cx-read=responses="$CX/bodies-ids3-on-read-r1/00001.req.json.gz" \
    oc-search=chat="$OC/00006.req.json.gz" cx-search=responses="$CX/bodies-ids3-on-search-r1/00001.req.json.gz" 2>&1 | tee "$D/replay.log"
  RC=${PIPESTATUS[0]}
  curl -s -m 5 -H 'authorization: Bearer goinfer' http://127.0.0.1:$PORT/admin/status > "$D/admin-status.json" 2>/dev/null
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
  [ "$RC" = 0 ] && [ -s "$D/replay.jsonl" ] || { echo "FAILED: arm $ARM replay rc=$RC"; exit 1; }
  # the pinned memory is gone with the process: nothing of this arm may remain
  echo "[$(date +%T)] arm $ARM done"
done
echo "finished $(date '+%F %T %Z')" >> "$OUT/provenance.txt"

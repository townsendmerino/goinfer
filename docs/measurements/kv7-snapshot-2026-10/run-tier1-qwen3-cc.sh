#!/bin/bash
# Tier 1 of the prefix snapshot, one Claude Code session start on Qwen3-4B (int4, f16 KV): two real Claude Code first requests, recorded
# through a stub with different task text in the same working directory (docs/measurements/side-requests-2026-10/anthropic_stub.py), replayed with snapshots on (the
# first cold, taking a snapshot of about 20K rows; the second restoring it after its own title request took the slot) and then off (the second only: a cold prefill of the same
# prompt, whose reply the restored one must equal).
#
#   B=$HOME/goinfer-bench/kv7-2026-10 bash run-tier1-qwen3-cc.sh      (under the timing lock: timing_lock.py run)
set -u
B=${B:?the directory holding bin/serve-cuda-kv7}
BIN=$B/bin/serve-cuda-kv7
OUT=${OUT:-$B/out-qwen3-cc}
MODEL=${MODEL:-$HOME/models/qwen3-4b}
PORT=${PORT:-18090}
S=$HOME/goinfer-bench/side-req-2026-10
HERE=$(cd "$(dirname "$0")" && pwd)
for f in "$BIN" "$MODEL/config.json" "$S/p3/req/00002.req.json.gz" "$S/p5/req/00002.req.json.gz" "$HERE/replay.py"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }
done
[ -z "$(ls "$HOME"/models | grep -E '\.v16\.giw$' )" ] || { echo "FATAL: a .v16.giw sidecar is under ~/models; this build writes v17"; exit 2; }
mkdir -p "$OUT" || exit 2
echo "binary $(cat "$BIN.rev") sha256 $(cat "$BIN.sha256")" > "$OUT/provenance.txt"
{ date '+start %F %T %Z'; uname -srm; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader; free -m | sed -n 1,3p; } >> "$OUT/provenance.txt"
for ARM in on off; do
  D=$OUT/$ARM; mkdir -p "$D"
  HB=auto; [ "$ARM" = off ] && HB=0
  # each session sends its title request (small, to the fast alias) a moment before its first main request, and takes the one slot
  SESS=(cc1t=messages="$S/p3/req/00001.req.json.gz@qwen3-4b-fast" cc1=messages="$S/p3/req/00002.req.json.gz"
        cc2t=messages="$S/p5/req/00001.req.json.gz@qwen3-4b-fast" cc2=messages="$S/p5/req/00002.req.json.gz")
  [ "$ARM" = off ] && SESS=(cc2t=messages="$S/p5/req/00001.req.json.gz@qwen3-4b-fast" cc2=messages="$S/p5/req/00002.req.json.gz")
  echo "[$(date +%T)] arm $ARM: starting the server (-kv-host-bytes $HB)"
  "$BIN" --model "$MODEL" --backend cuda --ctx 28672 --quant int4 --addr 127.0.0.1:$PORT --api-key goinfer --allow-admin \
    --log-requests --log-token-ids "$D/token-ids.jsonl" --kv-host-bytes "$HB" > "$D/serve.log" 2>&1 < /dev/null &
  PID=$!; echo $PID > "$D/serve.pid"
  for i in $(seq 1 240); do
    curl -s -m 2 -H 'authorization: Bearer goinfer' http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 && break
    kill -0 $PID 2>/dev/null || { echo "FATAL: the server exited during startup"; tail -5 "$D/serve.log"; exit 1; }
    sleep 2
  done
  curl -s -m 2 -H 'authorization: Bearer goinfer' http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 || { echo "FATAL: no server after 480 s"; kill $PID; exit 1; }
  grep -E "prefix snapshots|fit:|KV plan|KV f16|context:" "$D/serve.log" | cut -c1-200
  python3 -I "$HERE/replay.py" --base http://127.0.0.1:$PORT --pid $PID --out "$D" --max-tokens 24 --set model='"qwen3-4b"' "${SESS[@]}" 2>&1 | tee "$D/replay.log"
  RC=${PIPESTATUS[0]}
  curl -s -m 5 -H 'authorization: Bearer goinfer' http://127.0.0.1:$PORT/admin/status > "$D/admin-status.json" 2>/dev/null
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
  [ "$RC" = 0 ] && [ -s "$D/replay.jsonl" ] || { echo "FAILED: arm $ARM replay rc=$RC"; exit 1; }
  echo "[$(date +%T)] arm $ARM done"
done
echo "finished $(date '+%F %T %Z')" >> "$OUT/provenance.txt"

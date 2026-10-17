#!/bin/bash
# KV7 step 2, tier 1 (docs/tasks/task-kv7-step2-boundary-snapshot-2026-10.md §8): Claude Code session starts on Mellum2.1 (--ctx 32768, windowed KV, the
# expert cache, one slot). Two recorded first requests of one working directory (side-req-2026-10 p3 and p5, each preceded by its title request to the fast
# alias) and a compaction-shaped request rebuilt from the e2e transcripts (e2e-agents-2026-10-16/cc_rebuild.py, post1B), replayed with snapshots on and off.
# On: session 1 cold (takes the boundary snapshot), session 2 and the compaction-shaped request restore it. Off: session 2 only, cold (the do-nothing arm;
# one cold prefill is about 205 s, so the arm is not repeated for the third request: its reply is not compared, its restore and time to first token are read).
#
#   BIN=<serve-cuda binary> IN=<dir with p3 p5 post1B bodies> bash run-tier1-cc.sh        (under the timing lock: timing_lock.py run)
set -u
BIN=${BIN:?the serve binary built from the branch (CGO_ENABLED=0 -tags cuda cuda/cmd/serve)}
IN=${IN:?the directory of request bodies: p3-title p3-main p5-title p5-main post1B (.req.json.gz)}
OUT=${OUT:-$HOME/goinfer-bench/kv7-step2-2026-10/out-tier1}
MODEL=${MODEL:-$HOME/models/mellum2.1-thinking}
PORT=${PORT:-18090}
HERE=$(cd "$(dirname "$0")" && pwd)
REPLAY=${REPLAY:-$HERE/../kv7-snapshot-2026-10/replay.py}
for f in "$BIN" "$MODEL/config.json" "$IN/p3-title.req.json.gz" "$IN/p3-main.req.json.gz" "$IN/p5-title.req.json.gz" "$IN/p5-main.req.json.gz" "$IN/post1B.req.json.gz" "$REPLAY"; do
  [ -e "$f" ] || { echo "FATAL: $f is missing"; exit 2; }
done
[ -z "$(ls "$HOME"/models | grep -E '\.v16\.giw$' )" ] || { echo "FATAL: a .v16.giw sidecar is under ~/models; this build writes v17"; exit 2; }
mkdir -p "$OUT" || exit 2
echo "binary $(cat "$BIN.rev" 2>/dev/null) sha256 $(sha256sum "$BIN" | cut -d' ' -f1)" > "$OUT/provenance.txt"
{ date '+start %F %T %Z'; uname -srm; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader; free -m | sed -n 1,3p; } >> "$OUT/provenance.txt"
# The serve swap guard's threshold, 4096 MB as the e2e and I3 plans set it. At its default (512 MB) a first attempt (2026-10-17 07:02) was refused mid-replay
# after the kernel, at vm.swappiness 100, swapped 1.4 GB of the server's own pages while the model streamed through the page cache, before the snapshot
# was taken; the process's VmSwap is recorded in proc.csv either way and read in the record, not assumed flat.
export GOINFER_SWAP_GUARD=${GOINFER_SWAP_GUARD:-4096}
FAST='"mellum2.1-thinking-fast"'
# ARMS=cc3 runs only session 1 (cold, takes the snapshot) and the compaction-shaped request: the first run of that request used a body serialized with
# Python's default escaping, which rendered another prompt; the corrected body is built by cc_rebuild.py.
for ARM in ${ARMS:-on off}; do
  D=$OUT/$ARM; mkdir -p "$D"
  HB=auto; SESS=(cc1t=messages="$IN/p3-title.req.json.gz@mellum2.1-thinking-fast" cc1=messages="$IN/p3-main.req.json.gz"
                 cc2t=messages="$IN/p5-title.req.json.gz@mellum2.1-thinking-fast" cc2=messages="$IN/p5-main.req.json.gz"
                 cc3=messages="$IN/post1B.req.json.gz")
  if [ "$ARM" = cc3 ]; then SESS=(cc1t=messages="$IN/p3-title.req.json.gz@mellum2.1-thinking-fast" cc1=messages="$IN/p3-main.req.json.gz" cc3=messages="$IN/post1B.req.json.gz"); fi
  if [ "$ARM" = off ]; then HB=0; SESS=(cc2t=messages="$IN/p5-title.req.json.gz@mellum2.1-thinking-fast" cc2=messages="$IN/p5-main.req.json.gz"); fi
  echo "[$(date +%T)] arm $ARM: starting the server (-kv-host-bytes $HB)"
  "$BIN" --model "$MODEL" --backend cuda --ctx 32768 --moe-cache-experts --addr 127.0.0.1:$PORT --api-key goinfer --allow-admin \
    --log-requests --log-token-ids "$D/token-ids.jsonl" --kv-host-bytes "$HB" > "$D/serve.log" 2>&1 < /dev/null &
  PID=$!; echo $PID > "$D/serve.pid"
  for i in $(seq 1 240); do
    curl -s -m 2 -H 'authorization: Bearer goinfer' http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 && break
    kill -0 $PID 2>/dev/null || { echo "FATAL: the server exited during startup"; tail -5 "$D/serve.log"; exit 1; }
    sleep 2
  done
  curl -s -m 2 -H 'authorization: Bearer goinfer' http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 || { echo "FATAL: no server after 480 s"; kill $PID; exit 1; }
  grep -E "prefix snapshots|fit:|KV plan|windowed KV" "$D/serve.log" | cut -c1-200
  python3 -I "$REPLAY" --base http://127.0.0.1:$PORT --pid $PID --out "$D" --max-tokens 24 --set model='"mellum2.1-thinking"' "${SESS[@]}" 2>&1 | tee "$D/replay.log"
  RC=${PIPESTATUS[0]}
  curl -s -m 5 -H 'authorization: Bearer goinfer' http://127.0.0.1:$PORT/admin/status > "$D/admin-status.json" 2>/dev/null
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
  [ "$RC" = 0 ] && [ -s "$D/replay.jsonl" ] || { echo "FAILED: arm $ARM replay rc=$RC"; exit 1; }
  echo "[$(date +%T)] arm $ARM done"
done
echo "finished $(date '+%F %T %Z')" >> "$OUT/provenance.txt"

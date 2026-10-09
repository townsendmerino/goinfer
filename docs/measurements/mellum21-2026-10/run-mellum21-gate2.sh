#!/usr/bin/env bash
# Mellum2.1 Gate 2 (docs/tasks/task-mellum21-2026-10.md, "Gate 2 ... PRE-REGISTERED"): int4 on the 8 GB card, then the unmodified `serve check`.
# Run NOW at the owner's word (2026-10-09 "go ahead and run gate 2 now"); estimate ~40 min (int4 quantisation of a 24 GB bf16 checkpoint at load, check < 2 min).
# Registered flags: --model <ckpt> --backend cuda --ctx 16384; --moe-cache-experts ONLY if the plan declines a resident load (banner decode path not cuda). No flag is tuned after a reading.
# Leaves the server RUNNING (pid in $OUT/serve.pid) for the by-hand follow-ups the registration names; stop it with: kill "$(cat $OUT/serve.pid)"
set -uo pipefail
BIN=${BIN:-$HOME/goinfer-bench/s6fix/serve-cuda}
CK=${CK:-$HOME/models/mellum2.1-thinking}
PORT=${PORT:-18921}
OUT=${1:-$HOME/goinfer-logs/mellum21-gate2-$(date +%F)}
fatal() { echo "FATAL: $*" >&2; exit 2; }
[ -x "$BIN" ] || fatal "$BIN missing"; [ -f "$CK/config.json" ] || fatal "$CK missing"
case "$CK$OUT" in *"/srv/models"*|*"/Volumes/"*) fatal "an archive path";; esac
mkdir -p "$OUT"
{ echo "binary: $BIN $("$BIN" --version | head -1)"; echo "started: $(date '+%F %T %Z')"; echo "load: $(cat /proc/loadavg)"; free -g | sed -n 2p
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader; } | tee "$OUT/provenance.txt"
[ -n "${MELLUM_DRY:-}" ] && { echo "DRY ok"; exit 0; }
launch() { # $1 = tag, rest = extra flags
  local tag=$1; shift
  setsid nohup "$BIN" --model "$CK" --backend cuda --ctx 16384 --addr 127.0.0.1:$PORT "$@" </dev/null >"$OUT/serve-$tag.log" 2>&1 &
  echo $! > "$OUT/serve.pid"; echo "[$(date +%T)] launched $tag pid $(cat "$OUT/serve.pid") flags: $*"
}
waitready() { local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt 1800 ]; do
    curl -fs "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "[$(date +%T)] ready after $((SECONDS - t0))s"; return 0; }
    kill -0 "$(cat "$OUT/serve.pid")" 2>/dev/null || { echo "[$(date +%T)] server exited"; return 1; }
    [ $(( (SECONDS - t0) % 60 )) -lt 5 ] && echo "[$(date +%T)] ... loading $((SECONDS - t0))s; $(tail -n1 "$OUT/serve-$TAG.log" | cut -c1-140)"
    sleep 5
  done; echo "[$(date +%T)] not ready in 30 min"; return 1; }
TAG=plain; launch plain
PLAIN_OK=1; waitready || PLAIN_OK=0
grep -iE 'decode path|resident|VRAM|declin|KV|weights|cannot apply' "$OUT/serve-plain.log" | head -20 | tee "$OUT/banner-plain.txt"
if [ $PLAIN_OK = 0 ] || ! grep -q 'decode path: cuda' "$OUT/serve-plain.log"; then
  echo "[$(date +%T)] plain load did not come up resident on cuda -> registered fallback: --moe-cache-experts"
  kill "$(cat "$OUT/serve.pid")" 2>/dev/null; sleep 10
  TAG=moecache; launch moecache --moe-cache-experts
  waitready || { echo "RESULT: load failed (moe-cache)"; tail -20 "$OUT/serve-moecache.log"; exit 1; }
  grep -iE 'decode path|resident|VRAM|declin|KV|weights|C′' "$OUT/serve-moecache.log" | head -20 | tee "$OUT/banner.txt"
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee "$OUT/vram-idle.txt"
echo "[$(date +%T)] serve check"
timeout 900 "$BIN" check "http://127.0.0.1:$PORT" > "$OUT/check.log" 2>&1; echo "check rc=$?" | tee -a "$OUT/check.log"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee "$OUT/vram-after-check.txt"
cat "$OUT/check.log"
echo "finished: $(date '+%F %T %Z'); server left running, pid $(cat "$OUT/serve.pid")"

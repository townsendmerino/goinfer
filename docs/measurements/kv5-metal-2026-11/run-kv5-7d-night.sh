#!/bin/bash
# KV5-7d's timed readings, as registered (docs/tasks/task-kv-architecture-2026-10.md §6, KV5-7's registration, G2 (7d)), for the night queue:
#   1. gemma3-1b decode at 128, 2,048 and 3,900, five A B B A quads, old against new;
#   2. gemma3-1b prefill TTFT at K = 512 and 3,900, three alternating pairs (ttft_ab.py);
#   3. the 1.5B control, decode at 2,048, five quads (it has no window group; each layer now binds a table of its own);
#   4. MC3 at 4 clients on the 1.5B, once (run_mc3_kv5_7.py).
# The 4 depth-128 cells of the earlier attempts were all void ("swap-out count unreadable"), so no cell is carried over; the run starts from none.
# old: f142c8e4e (main just before KV5-7a); new: 45ef408a7 (KV5-7c-ii's code commit). Prebuilt serve binaries and a copy of the harness, so the
# tree may move: run this script from a pinned archive of the commit that holds it (the queue command names it).
# Battery no longer voids a cell for this reading (amended 2026-11-15, before any cell ran): every stage records its power source and battery
# percentage, Low Power Mode must be off, and the run stops cleanly under 20%; the idle gate, heavy load, swap and the exclusive lock stand.
# Exits 75 when the box refuses a stage for load or memory, so night.py retries; a finished stage leaves STAGE.done and is skipped, and the
# harness resumes the cells it had done. Exits 1 when a stage failed for another reason, after trying the rest.
# `run-kv5-7d-night.sh --check` runs only the file checks (and the .gguf to .giw mapping) and exits 0 or 1: run it when queuing, so a missing file
# shows at once and not at the job's start. ~/models keeps the Metal .giw bundles, not the .gguf files, and serve exits on an absent .gguf, so the
# harness's serve binaries are launchers (serve-giw-wrap.sh) that map each absent .gguf to the .giw beside it; MC3 is given the .giw directly.
P="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$HOME/goinfer-work/kv5-7"
OLDB="$D/serve-bis-f142c8e4e"; NEWB="$D/serve-bis-45ef408a7"
OUT="$D/night"; mkdir -p "$OUT/serve-logs"
WRAP="$P/docs/measurements/kv5-metal-2026-11/serve-giw-wrap.sh"
GEMMA="$HOME/models/gemma3-1b-q4_k_m.int4.metal.v17.giw"; Q15="$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw"
need=0
for f in "$OLDB" "$NEWB" "$WRAP" "$P/scripts/bench_peer.py" "$P/scripts/bench_w7_plain.py" "$P/scripts/bench_peer_prefill.py" "$GEMMA" "$Q15" "$GEMMA.verified" "$Q15.verified"; do
  [ -e "$f" ] || { echo "missing $f"; need=1; }
done
# the launcher's mapping, on a name that is absent: an absent .gguf beside a present .giw becomes the .giw
got=$(SERVE_WRAP_REAL=/bin/echo "$WRAP" -model "bench=$HOME/models/gemma3-1b-q4_k_m.gguf" 2>&1)
[ "$got" = "-model bench=$GEMMA" ] || { echo "launcher maps to '$got', want '-model bench=$GEMMA'"; need=1; }
if [ "${1:-}" = "--check" ]; then [ "$need" = 0 ] && echo "check: all files present, launcher maps" ; exit "$need"; fi
[ "$need" = 0 ] || exit 1
# one launcher per serve binary: the harness runs each with its own path
for v in old new; do
  b=$OLDB; [ "$v" = new ] && b=$NEWB
  printf '#!/bin/bash\nexport SERVE_WRAP_REAL=%s\nexec %s "$@"\n' "$b" "$WRAP" > "$OUT/serve-$v.sh"; chmod +x "$OUT/serve-$v.sh"
done
power() { pmset -g batt | head -2 | tail -1 | sed 's/^ *//'; }
pct() { pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%'; }
echo "old f142c8e4e, new 45ef408a7, harness $P"
date '+start %H:%M'; uptime; echo "power: $(pmset -g batt | head -1); $(power)"; pmset -g | grep -i lowpowermode
[ "$(pmset -g | grep -i lowpowermode | awk '{print $2}')" = "0" ] || { echo "Low Power Mode is on: not running, not changing it"; exit 75; }
export BENCH_DEPTH_BACKEND=metal BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=metal BENCH_REQUIRE_BACKEND=1
export BENCH_SERVE_LOG_DIR="$OUT/serve-logs" BENCH_LOAD_TIMEOUT=420 GOINFER_SERVE_METAL="$OUT/serve-new.sh" GOINFER_SERVE_METAL_OLD="$OUT/serve-old.sh"
export BENCH_SWAPOUT_VOID_MB=0 BENCH_SWAP_VOID_STOP=3 BENCH_IDLE_WAIT=300
unset BENCH_SWAPIN_VOID_MB
# the power sampler: a line every 30 s, and a stop under 20%
( while true; do
    echo "$(date +%H:%M:%S) $(pmset -g batt | head -1 | sed 's/Now drawing from //') | $(power) | load1 $(uptime | sed 's/.*load averages: //' | awk '{print $1}')" >> "$OUT/power.log"
    p=$(pct)
    if [ -n "$p" ] && [ "$p" -lt 20 ]; then
      echo "battery $p% under 20%: stopping" >> "$OUT/power.log"; touch "$OUT/STOP"
      [ -s "$OUT/current.pid" ] && kill -TERM "$(cat "$OUT/current.pid")" 2>/dev/null
    fi
    sleep 30
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
failed=0; refused=0
stage() { # name cmd...
  name=$1; shift
  [ -e "$OUT/$name.done" ] && { echo "== $name: done before, skipped"; return; }
  [ -e "$OUT/STOP" ] && return
  date "+== $name %H:%M:%S; $(power)"
  "$@" > "$OUT/$name.log" 2>&1 &
  echo $! > "$OUT/current.pid"
  wait $!
  rc=$?
  : > "$OUT/current.pid"
  date "+== $name rc=$rc %H:%M:%S; $(power)"
  if [ "$rc" = 0 ]; then : > "$OUT/$name.done"; return; fi
  tail -1 "$OUT/$name.log" | cut -c1-200 | sed 's/VOID/void/g'
  if grep -q '^REFUSED' "$OUT/$name.log"; then refused=1; else failed=1; fi
}
cd "$P/scripts" || exit 1
stage g2-gemma env BENCH_ABBA=5 BENCH_MODELS=gemma3-1b BENCH_DEPTHS=2048,3900 python3 -u bench_peer.py "$OUT/g2-gemma3-1b.json"
stage ttft-gemma env TTFT_MODEL=gemma3-1b TTFT_SCRIPTS="$P/scripts" python3 -u "$P/docs/measurements/kv5-metal-2026-11/ttft_ab.py" "$OLDB" "$NEWB" "$OUT/ttft-gemma3-1b.json"
stage control-1.5b env BENCH_ABBA=5 BENCH_MODELS=1.5B BENCH_PHASES=B BENCH_DEPTHS=2048 python3 -u bench_peer.py "$OUT/g2-control-1.5b.json"
stage mc3 env BENCH_W7_MODEL="$Q15" python3 -u "$P/docs/measurements/kv5-metal-2026-11/run_mc3_kv5_7.py" "$OLDB" "$NEWB" "$P/scripts" "$OUT/mc3"
date '+end %H:%M'
if [ -e "$OUT/STOP" ]; then echo "STOPPED: battery under 20%; what ran is in $OUT"; exit 1; fi
[ "$failed" = 0 ] || { echo "a stage failed: see $OUT/*.log"; exit 1; }
[ "$refused" = 0 ] || { echo "a stage was refused (load or memory): retry resumes it"; exit 75; }
echo "all stages done"

#!/bin/bash
# Step-flag memory, Nemotron-H 4B's arms (docs/measurements/step-flags-2026-11/README.md): the fit guard refused them by day. Pinned test binary,
# serve and harness. Serve writes the Metal bundle once, then every arm loads from it; a direct safetensors load quantizes in memory and swaps.
# Exits 75 when the box lacks memory, so night.py retries; exits 1 on swap in the first load.
D="$HOME/goinfer-work/step-flags"
P="$HOME/goinfer-work/kv5-2-g2"
M="$HOME/models/nemotron-h-4b-instruct-128k"
L="$D/nemotron-4b-first-load.log"
cd "$D" || exit 1
for f in "$D/metal.test" "$P/serve-new" "$M/config.json"; do
  [ -e "$f" ] || { echo "missing $f"; exit 1; }
done
pmset -g batt | head -1 | grep -q 'AC Power' || { echo "on battery: refused"; exit 75; }
r=$(python3 "$P/scripts/hostmem.py" --reclaimable)
date "+start %H:%M:%S; reclaimable $r MB; load $(sysctl -n vm.loadavg | cut -c3-7)"
[ "${r:-0}" -ge 7000 ] || { echo "reclaimable under 7 GB: refused"; exit 75; }
swap() { vm_stat | awk '/^Swapins:/{i=$2+0} /^Swapouts:/{o=$2+0} END{print i, o}'; }
G=$(ls "$HOME/models"/nemotron-h-4b-instruct-128k*.metal*.giw 2>/dev/null | head -1)
if [ -z "$G" ]; then
  read -r i0 o0 < <(swap)
  "$P/serve-new" -model "bench=$M" -backend metal -addr 127.0.0.1:18099 -ctx 4096 -quant int4 -require-backend > "$L" 2>&1 &
  pid=$!
  for _ in $(seq 1 900); do
    curl -s -m 1 http://127.0.0.1:18099/health > /dev/null && break
    kill -0 "$pid" 2>/dev/null || break
    /bin/sleep 1
  done
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  read -r i1 o1 < <(swap)
  date "+first load done %H:%M:%S; swap-ins $((i1-i0)) pages, swap-outs $((o1-o0)) pages"
  grep -E 'fit:|KV plan' "$L" | cut -c1-160
  G=$(ls "$HOME/models"/nemotron-h-4b-instruct-128k*.metal*.giw 2>/dev/null | head -1)
  [ -n "$G" ] || { echo "no Metal bundle written"; tail -3 "$L" | cut -c1-200; exit 75; }
  [ "$((o1-o0))" -le 6400 ] || { echo "over 100 MB of swap-outs in the first load: stopping"; exit 1; }
fi
read -r i0 o0 < <(swap)
STEPFLAGS_PATH="$G" STEPFLAGS_FLAG=recurrent STEPFLAGS_OUT="$D/nemotron-4b.jsonl" "$D/metal.test" -test.count=1 -test.v -test.timeout 30m -test.run '^TestStepFlagsMemory$' > "$D/nemotron-4b.log" 2>&1
rc=$?
read -r i1 o1 < <(swap)
date "+arms done %H:%M:%S rc=$rc; swap-ins $((i1-i0)) pages, swap-outs $((o1-o0)) pages"
grep -E '^--- ' "$D/nemotron-4b.log" | cut -c1-80
exit "$rc"

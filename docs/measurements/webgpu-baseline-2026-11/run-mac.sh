#!/usr/bin/env bash
# WG0's Mac half, registered in registration.md (this directory, § "The Mac half") before any timed run: WebGPU (over Metal)
# against goinfer's Metal backend, the 0.5B and 1.5B. The same shape as run-nobara.sh with the macOS differences the
# registration names: no page-cache eviction (no posix_fadvise), the darwin idle gate, the settle cap at load 2.0.
#   bash run-mac.sh             # fit checks, longest-context probes, then the decode steps and the TTFT step
#   FIT_ONLY=1 bash run-mac.sh  # the fit checks and probes alone
#   bash run-mac.sh <step,step> # only the named steps
# Run it as: python3 <pinned wt>/scripts/timing_lock.py run --wait 3600 --label wg0-mac -- bash run-mac.sh
# Both serve binaries are pre-built at $REV from the clean detached worktree $H and checked by sha256 (SHA_M / SHA_W, set by
# the caller from the build); bench_peer.py runs from $H.
set -uo pipefail
STEPS=${1:-}
export PATH=/opt/homebrew/bin:/usr/local/go/bin:$PATH
HERE=$(cd "$(dirname "$0")" && pwd)
REV=018cbe74
W=$HOME/goinfer-bench/wg0-mac
H=$W/wt
SM=$W/bin/serve-metal-$REV
SW=$W/bin/serve-webgpu-$REV
: "${SHA_M:?set SHA_M to the serve-metal build hash}" "${SHA_W:?set SHA_W to the serve-webgpu build hash}"
OUT=$W/results
L=$OUT/fit
mkdir -p "$OUT" "$L"
ts() { date '+%F %T %Z'; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'; }
state() { echo "load $(sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1,$2,$3}') free $(memory_pressure | awk -F': ' '/free percentage/{print $2}') swap $(swap_mb) MB power $(pmset -g batt | head -1 | sed "s/.*'\(.*\)'.*/\1/")"; }

[ "$(git -C "$H" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "FATAL: $H is not at $REV"; exit 2; }
[ -z "$(git -C "$H" status --porcelain --untracked-files=no)" ] || { echo "FATAL: $H has tracked changes"; exit 2; }
echo "$SHA_M  $SM" | shasum -a 256 -c --quiet || { echo "FATAL: $SM is not the recorded build"; exit 2; }
echo "$SHA_W  $SW" | shasum -a 256 -c --quiet || { echo "FATAL: $SW is not the recorded build"; exit 2; }
M=$HOME/models
P05=$M/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf P15=$M/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
for p in $P05 $P15; do
  r=$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$p")
  case "$r" in /srv/models*|/Volumes/*) echo "FATAL: $p resolves to the archive ($r)"; exit 2;; esac
  [ -f "$r" ] || { echo "FATAL: $p is missing"; exit 2; }
done
{ echo "started   $(ts)"; echo "rev       $REV ($H, clean); scripts from $HERE"
  echo "serve     $SM sha256 $SHA_M"; echo "serve     $SW sha256 $SHA_W"
  echo "host      $(hostname) $(uname -sr) macOS $(sw_vers -productVersion)"
  echo "gpu       $(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model|Total Number of Cores/{printf "%s; ", $2}')"
  echo "aikit     $(awk '/townsendmerino\/aikit /{print $2}' "$H/go.mod")"
  echo "therm     $(pmset -g therm 2>/dev/null | grep -v '^$' | tr '\n' ';')"
  echo "lock      $(python3 "$H/scripts/timing_lock.py" status 2>&1 | head -2 | tr '\n' ' ')"
  echo "steps     ${STEPS:-all}"; echo "state     $(state)"; } | tee -a "$OUT/provenance.txt"

fit() { # fit <name> <want-decode-path-prefix> <serve> <args...>
  local name=$1 want=$2 bin=$3; shift 3
  local log=$L/$name.log
  "$bin" "$@" -addr 127.0.0.1:18197 > "$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 600); do grep -q 'goinfer serving on' "$log" && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  if ! grep -q 'goinfer serving on' "$log"; then echo "FIT REFUSED $name: $(grep -iE 'refus|declin|not fit|error' "$log" | head -2 | tr '\n' ' ')"; return 1; fi
  local dp; dp=$(grep -m1 'decode path:' "$log" | sed 's/^ *decode path: //')
  case "$dp" in "$want"*) ;; *) echo "FIT WRONG PATH $name: decode path '$dp', want '$want'"; return 1;; esac
  echo "fit $name: $dp | $(grep -m1 ' fit:' "$log" | sed 's/^ *//') | $(grep -m1 'KV plan:' "$log" | sed 's/^ *//')"
}
echo "== fit checks $(ts) $(state)"
fok=0
for m in 0.5B:$P05 1.5B:$P15; do k=${m%%:*}; p=${m#*:}
  fit metal-$k metal-resident "$SM" -model bench=$p -backend metal -ctx 4096 -quant int4 -require-backend || fok=1
  # the WebGPU arm exactly as the harness launches it: its -backend metal, then BENCH_GOINFER_OLD_ARGS' -backend webgpu
  fit webgpu-$k webgpu:metal-resident "$SW" -model bench=$p -backend metal -ctx 4096 -quant int4 -require-backend -backend webgpu || fok=1
  fit webgpu-ttft-$k webgpu:metal-resident "$SW" -model bench=$p -backend webgpu -ctx 4096 -quant int4 -require-backend || fok=1
done
[ $fok = 0 ] || { echo "== a fit check refused or resolved the wrong path: nothing timed; exit 75 $(ts)"; exit 75; }
echo "== fit checks done $(ts)"

# Longest context that fits (the registration's amendment of 2026-11-20): descriptive, no gate, untimed; the same probes as
# run-nobara.sh with Metal in CUDA's place.
ctxfit() { # ctxfit <name> <want> <serve> <args...>: writes the banner lines to $L/ctxfit-<name>.log and one row to ctxfit.tsv
  local name=$1 want=$2 bin=$3; shift 3
  local log=$L/ctxfit-$name.log
  "$bin" "$@" -addr 127.0.0.1:18197 > "$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 600); do grep -q 'goinfer serving on' "$log" && break; kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  local dp kv fitl
  dp=$(grep -m1 'decode path:' "$log" | sed 's/^ *decode path: //')
  kv=$(grep -m1 'KV plan:' "$log" | sed 's/^ *//')
  fitl=$(grep -m1 -E ' fit:|resident KV slots' "$log" | sed 's/^ *//')
  if grep -q 'goinfer serving on' "$log" && [[ "$dp" == "$want"* ]]; then
    printf '%s\tok\t%s\t%s\t%s\t%s\n' "$name" "$*" "$dp" "$kv" "$fitl" >> "$OUT/ctxfit.tsv"; echo "ctxfit $name: ok | $dp | $kv | $fitl"; return 0
  fi
  printf '%s\tno\t%s\t%s\t%s\t%s\n' "$name" "$*" "$dp" "$(grep -iE -m2 'refus|declin|not fit|error|failed' "$log" | tr '\n' ' ')" "$fitl" >> "$OUT/ctxfit.tsv"
  echo "ctxfit $name: no | $dp | $(grep -iE -m1 'refus|declin|not fit|error|failed' "$log")"; return 1
}
if [ ! -s "$OUT/ctxfit.tsv" ]; then
  echo "== longest-context probes $(ts) $(state)"
  for m in 0.5B:$P05 1.5B:$P15; do k=${m%%:*}; p=${m#*:}
    ctxfit metal-$k-default metal-resident "$SM" -model bench=$p -backend metal -quant int4
    ctxfit webgpu-$k-default webgpu:metal-resident "$SW" -model bench=$p -backend webgpu -quant int4
    for c in 32768 28672 24576 20480 16384 12288 8192 4096; do
      ctxfit metal-$k-one-$c metal-resident "$SM" -model bench=$p -backend metal -quant int4 -kv-sessions 1 -ctx $c -require-backend && break
    done
    for c in 32768 16384 12288 8192 4096; do
      ctxfit webgpu-$k-one-$c webgpu:metal-resident "$SW" -model bench=$p -backend webgpu -quant int4 -kv-sessions 1 -ctx $c -require-backend && break
    done
  done
  echo "== longest-context probes done $(ts)"
fi
[ "${FIT_ONLY:-}" = 1 ] && exit 0

rc_all=0
settle() {
  local t0 la; t0=$(date +%s)
  while :; do
    la=$(sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1}')
    awk -v l="$la" 'BEGIN{exit !(l<2.0)}' && { echo "=== settle: load $la < 2.0 after $(( $(date +%s) - t0 ))s"; return 0; }
    [ $(( $(date +%s) - t0 )) -ge 600 ] && { echo "=== settle: load $la still >= 2.0 after 600s"; return 1; }
    sleep 20
  done
}
step() { # step <name> <results.json> <command...>
  local name=$1 res=$2; shift 2
  if [ -n "$STEPS" ] && [[ ",$STEPS," != *",$name,"* ]]; then echo "=== skip $name"; return 0; fi
  [ -e "$OUT/$name.log" ] && mv "$OUT/$name.log" "$OUT/$name.attempt-$(date -r "$OUT/$name.log" +%Y%m%dT%H%M%S).log"
  echo "=== no page-cache eviction on macOS $(state)"
  if ! settle; then echo "FAILED step $name: the load never settled"; rc_all=1; return 0; fi
  echo "=== $(ts) START $name $(state)"
  local t0; t0=$(date +%s)
  (cd "$H" && BENCH_SERVE_LOG_DIR="$OUT/serve-logs" "$@") > "$OUT/$name.log" 2>&1
  local r=$?
  if [ $r -eq 0 ] && grep -q '^REFUSED\|REFUSED mid-sweep\|^STOPPED' "$OUT/$name.log"; then r=3; fi
  if [ $r -eq 0 ] && [ ! -s "$res" ]; then r=4; fi
  echo "=== $(ts) END $name rc=$r ($(( $(date +%s) - t0 ))s) $(state)"
  if [ $r -ne 0 ]; then echo "FAILED step $name rc=$r"; tail -5 "$OUT/$name.log"; rc_all=1; fi
}
# native = engine goinfer (serve-metal, -backend metal); WebGPU = engine goinfer_old (serve-webgpu; its -backend webgpu comes after
# the harness's -backend metal, and the later flag wins). BENCH_ABBA=2: per cell, blocks native,WebGPU / WebGPU,native, twice.
export GOINFER_SERVE_METAL=$SM GOINFER_SERVE_METAL_OLD=$SW BENCH_GOINFER_OLD_ARGS="-backend webgpu"
export BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=metal BENCH_DEPTH_BACKEND=metal BENCH_PHASES=B BENCH_ABBA=2 BENCH_RUNS=2
export BENCH_REQUIRE_BACKEND=1 BENCH_SWAP_VOID_STOP=3 BENCH_SWAP_KILL_MB=1024
for mk in 0.5B 1.5B; do for d in 128 2048; do
  step decode-$mk-$d "$OUT/decode-$mk-$d.json" env BENCH_MODELS=$mk BENCH_DEPTHS=$d python3 -u scripts/bench_peer.py "$OUT/decode-$mk-$d.json"
done; done
step ttft "$OUT/ttft.json" env WG0_REPO=$H WG0_SETTLE_LOAD=2.0 python3 -u "$HERE/wg0_ttft.py" "$OUT/ttft.json" \
  --arm metal=$SM:metal --arm webgpu=$SW:webgpu --models 0.5B,1.5B --quads 2 --n 6 --ctx 4096
python3 "$HERE/grade.py" "$OUT" | tee "$OUT/grade.txt"
echo "finished  $(ts) rc=$rc_all $(state)" | tee -a "$OUT/provenance.txt"
exit $rc_all

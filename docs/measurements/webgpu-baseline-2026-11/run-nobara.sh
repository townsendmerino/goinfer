#!/usr/bin/env bash
# WG0's nobara half, registered in registration.md (this directory) before any timed run.
#   bash run-nobara.sh            # fit checks, then the decode steps and the TTFT step, under the caller's exclusive lock
#   FIT_ONLY=1 bash run-nobara.sh # the fit checks alone
#   bash run-nobara.sh <step,step> # only the named steps (re-run of steps that produced no results)
# Run it as: python3 <pinned wt>/scripts/timing_lock.py run --wait 7200 --label wg0 -- bash run-nobara.sh
# Both serve binaries are pre-built at $REV from the clean detached worktree $H and checked by sha256; bench_peer.py runs from $H.
# Before each step it evicts the checkpoints' page cache (evict_cache.py) and waits for the 1-minute load to settle under 0.8;
# wg0_ttft.py evicts and settles before each of its cells.
set -uo pipefail
STEPS=${1:-}
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REV=018cbe74
DRIVER=595.112.07
W=$HOME/goinfer-bench/wg0-run
H=$W/wt
SC=$W/bin/serve-cuda-$REV
SW=$W/bin/serve-webgpu-$REV
SHA_C=778850e2bf22cea6b92673c63a155767381acd19d6c9b30d0122ad98ebcb9fb2
SHA_W=7e1c0a137413519ccc80dfc2d70e9217b62a1508dfc34f13bd8169e4a699a41d
OUT=$W/results
L=$OUT/fit
mkdir -p "$OUT" "$L"
ts() { date '+%F %T %Z'; }
state() { echo "load $(cut -d' ' -f1-3 /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB swap $(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo) MiB"; }

[ "$(git -C "$H" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "FATAL: $H is not at $REV"; exit 2; }
[ -z "$(git -C "$H" status --porcelain --untracked-files=no)" ] || { echo "FATAL: $H has tracked changes"; exit 2; }
echo "$SHA_C  $SC" | sha256sum -c --quiet || { echo "FATAL: $SC is not the registered build"; exit 2; }
echo "$SHA_W  $SW" | sha256sum -c --quiet || { echo "FATAL: $SW is not the registered build"; exit 2; }
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
[ "$drv" = "$DRIVER" ] || { echo "FATAL: driver is $drv, registered $DRIVER"; exit 2; }
M=$HOME/models
P05=$M/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf P15=$M/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf P7=$M/qwen2.5-7b-instruct-q4_k_m.gguf
for p in $P05 $P15 $P7; do
  r=$(readlink -f "$p"); case "$r" in /srv/models*|/Volumes/*) echo "FATAL: $p resolves to the archive ($r)"; exit 2;; esac
  [ -e "$r" ] || { echo "FATAL: $p is missing"; exit 2; }
done
{ echo "started   $(ts)"; echo "rev       $REV ($H, clean); scripts from $HERE ($(git -C "$HERE" rev-parse --short=9 HEAD))"
  echo "serve     $SC sha256 $SHA_C"; echo "serve     $SW sha256 $SHA_W"
  echo "host      $(hostname) $(uname -sr)"; echo "gpu       $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"
  echo "vulkan    $(vulkaninfo --summary 2>/dev/null | grep -m2 -E 'deviceName|driverInfo' | tr -s ' ' | tr '\n' ';')"
  echo "aikit     $(awk '/townsendmerino\/aikit /{print $2}' "$H/go.mod")"
  echo "desktop   $(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | tr '\n' ';')"
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
for m in 0.5B:$P05 1.5B:$P15 7B:$P7; do k=${m%%:*}; p=${m#*:}
  fit cuda-$k cuda-resident "$SC" -model bench=$p -backend cuda -ctx 4096 -quant int4 -require-backend || fok=1
  # the WebGPU arm exactly as the harness launches it: its -backend cuda, then BENCH_GOINFER_OLD_ARGS' -backend webgpu
  fit webgpu-$k webgpu:vulkan-resident "$SW" -model bench=$p -backend cuda -ctx 4096 -quant int4 -require-backend -backend webgpu || fok=1
  fit webgpu-ttft-$k webgpu:vulkan-resident "$SW" -model bench=$p -backend webgpu -ctx 4096 -quant int4 -require-backend || fok=1
done
[ $fok = 0 ] || { echo "== a fit check refused or resolved the wrong path: nothing timed; exit 75 $(ts)"; exit 75; }
echo "== fit checks done $(ts)"
[ "${FIT_ONLY:-}" = 1 ] && exit 0

rc_all=0
settle() {
  local t0 la; t0=$(date +%s)
  while :; do
    la=$(cut -d' ' -f1 /proc/loadavg)
    awk -v l="$la" 'BEGIN{exit !(l<0.8)}' && { echo "=== settle: load $la < 0.8 after $(( $(date +%s) - t0 ))s"; return 0; }
    [ $(( $(date +%s) - t0 )) -ge 600 ] && { echo "=== settle: load $la still >= 0.8 after 600s"; return 1; }
    sleep 20
  done
}
step() { # step <name> <results.json> <command...>
  local name=$1 res=$2; shift 2
  if [ -n "$STEPS" ] && [[ ",$STEPS," != *",$name,"* ]]; then echo "=== skip $name"; return 0; fi
  [ -e "$OUT/$name.log" ] && mv "$OUT/$name.log" "$OUT/$name.attempt-$(date -r "$OUT/$name.log" +%Y%m%dT%H%M%S).log"
  echo "=== evict: $(python3 "$HERE/evict_cache.py" $P05 $P15 $P7 2>&1) $(state)"
  if ! settle; then echo "FAILED step $name: the load never settled"; rc_all=1; return 0; fi
  echo "=== $(ts) START $name $(state)"
  local t0; t0=$(date +%s)
  (cd "$H" && BENCH_SERVE_LOG_DIR="$OUT/serve-logs" "$@") > "$OUT/$name.log" 2>&1
  local r=$?
  if [ $r -eq 0 ] && grep -q '^REFUSED\|REFUSED mid-sweep\|^STOPPED' "$OUT/$name.log"; then r=3; fi
  if [ $r -eq 0 ] && [ ! -s "$res" ]; then r=4; fi
  echo "=== $(ts) END $name rc=$r ($(( $(date +%s) - t0 ))s)"
  if [ $r -ne 0 ]; then echo "FAILED step $name rc=$r"; tail -5 "$OUT/$name.log"; rc_all=1; fi
}
# native = engine goinfer (serve-cuda, -backend cuda); WebGPU = engine goinfer_old (serve-webgpu; its -backend webgpu comes after the
# harness's -backend cuda, and the later flag wins). BENCH_ABBA=2: per cell, blocks native,WebGPU / WebGPU,native, twice.
export GOINFER_SERVE_CUDA=$SC GOINFER_SERVE_CUDA_OLD=$SW BENCH_GOINFER_OLD_ARGS="-backend webgpu"
export BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_DEPTH_BACKEND=cuda BENCH_PHASES=B BENCH_ABBA=2 BENCH_RUNS=2
export BENCH_REQUIRE_BACKEND=1 BENCH_SWAP_VOID_STOP=3 BENCH_SWAP_KILL_MB=1024
for mk in 0.5B 1.5B 7B; do for d in 128 2048; do
  step decode-$mk-$d "$OUT/decode-$mk-$d.json" env BENCH_MODELS=$mk BENCH_DEPTHS=$d python3 -u scripts/bench_peer.py "$OUT/decode-$mk-$d.json"
done; done
step ttft "$OUT/ttft.json" env WG0_REPO=$H python3 -u "$HERE/wg0_ttft.py" "$OUT/ttft.json" --arm cuda=$SC:cuda --arm webgpu=$SW:webgpu \
  --models 0.5B,1.5B,7B --quads 2 --n 6 --ctx 4096
python3 "$HERE/grade.py" "$OUT" | tee "$OUT/grade.txt"
echo "finished  $(ts) rc=$rc_all $(state)" | tee -a "$OUT/provenance.txt"
exit $rc_all

#!/usr/bin/env bash
# The nobara peer re-anchor sweep, registered in docs/measurements/peer-reanchor-2026-11-19/README.md before any data.
#   bash run-nobara.sh cuda   # cells a, b, f, d, c (CUDA); its own night job, queued first
#   bash run-nobara.sh cpu    # cells e and v (x86 CPU); its own night job, queued second
#   FIT_ONLY=1 bash run-nobara.sh <part>   # the fit checks alone, no timing
#   bash run-nobara.sh <part> <step,step>  # only the named steps (a re-run of the steps that produced no results)
# Before each step it waits for the 1-minute load average to settle under SETTLE_LOAD (0.8) for up to 10 minutes, so the
# harness's one-sample preflight (BENCH_MAX_LOADAVG=1.0, unchanged) does not refuse on the load the fit checks or the
# previous step's servers left behind; a load that never settles fails the step as before.
# Before that wait it drops the page cache of every checkpoint file (evict_cache.py: posix_fadvise DONTNEED over ~/models
# and Ollama's store, no root, no setting changed), so a step's loads start with free memory instead of a full page cache,
# where the kernel pushes idle processes to zram and the void rule's zero swap bar voids the cell (README, "Why swap grows").
# Serve binaries are pre-built from a clean detached worktree at $REV and checked by sha256; bench_peer.py runs from that
# same worktree. Every goinfer load the part makes is fit-checked first with the binary and flags the harness passes; a
# refusal exits 75 (night.py re-queues it until --retry-until). Exits non-zero when any step fails or writes no results.
set -uo pipefail
PART=${1:?usage: run-nobara.sh cuda|cpu [step,step...]}
STEPS=${2:-}
case $PART in cuda|cpu) ;; *) echo "unknown part $PART"; exit 2;; esac
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REV=e29da447
DRIVER=595.112.07
W=$HOME/goinfer-bench/peer-reanchor-2026-11-19
H=$W/wt
SCUDA=$W/bin/serve-cuda-$REV
SCPU=$W/bin/serve-cpu-$REV
SHA_CUDA=703b01ed968be7d860bf68f76bdcc5ba8169ff3bb5cf8166f3d9234eaf2e77b8
SHA_CPU=27b18a32402f07edb07e8c588efb374dc8be64747d3a0d318e90a09283197791
OUT=$W/results/$PART
L=$OUT/fit
mkdir -p "$OUT" "$L"
ts() { date '+%F %T %Z'; }
state() { echo "load $(cut -d' ' -f1-3 /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB swap $(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo) MiB"; }

# --- pins -------------------------------------------------------------------------------------------------------------
[ "$(git -C "$H" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "FATAL: $H is not at $REV"; exit 2; }
[ -z "$(git -C "$H" status --porcelain --untracked-files=no)" ] || { echo "FATAL: $H has tracked changes"; exit 2; }
echo "$SHA_CUDA  $SCUDA" | sha256sum -c --quiet || { echo "FATAL: $SCUDA is not the registered build"; exit 2; }
echo "$SHA_CPU  $SCPU" | sha256sum -c --quiet || { echo "FATAL: $SCPU is not the registered build"; exit 2; }
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
[ "$drv" = "$DRIVER" ] || { echo "FATAL: driver is $drv, registered $DRIVER: a re-anchor, not a carry-forward. Stopping."; exit 2; }
M=$HOME/models
P05=$M/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf P15=$M/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf P7=$M/qwen2.5-7b-instruct-q4_k_m.gguf
PG1=$M/gemma3-1b-q4_k_m.gguf PPHI=$M/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
PM26=$M/gemma4-26b-int4.giw PM26PEER=$M/gemma4-26b-q4_k_m.gguf PG20=$M/gpt-oss-20b-MXFP4.gguf
for p in $P05 $P15 $P7 $PG1 $PPHI $PM26 $PM26PEER $PG20; do
  r=$(readlink -f "$p"); case "$r" in /srv/models*|/Volumes/*) echo "FATAL: $p resolves to the archive ($r)"; exit 2;; esac
  [ -e "$r" ] || { echo "FATAL: $p is missing"; exit 2; }
done
{ echo "part      $PART"; echo "started   $(ts)"; echo "rev       $REV ($H, clean)"
  echo "serve     $SCUDA sha256 $SHA_CUDA"; echo "serve     $SCPU sha256 $SHA_CPU"
  echo "host      $(hostname) $(uname -sr)"; echo "gpu       $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"
  echo "aikit     $(awk '/townsendmerino\/aikit /{print $2}' "$H/go.mod")"
  echo "ollama    $("$HOME/ollama-0325/bin/ollama" --version 2>&1 | tail -1)"
  echo "llama     $(LD_LIBRARY_PATH= "$HOME/mycode/peers/llama.cpp/build/bin/llama-server" --version 2>&1 | head -1)"
  echo "desktop   $(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | tr '\n' ';')"
  echo "steps     ${STEPS:-all}"
  echo "state     $(state)"; } | tee -a "$OUT/provenance.txt"

# --- fit checks ---------------------------------------------------------------------------------------------------------
fit() { # fit <name> <want-decode-path-prefix> <serve> <args...>: the server's own banner, or a refusal
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
if [ "$PART" = cuda ]; then
  for c in 4096 8192; do
    fit 0.5B-ctx$c cuda-resident "$SCUDA" -model bench=$P05 -backend cuda -ctx $c -quant int4 || fok=1
    fit 1.5B-ctx$c cuda-resident "$SCUDA" -model bench=$P15 -backend cuda -ctx $c -quant int4 || fok=1
    fit 7B-ctx$c cuda-resident "$SCUDA" -model bench=$P7 -backend cuda -ctx $c -quant int4 || fok=1
  done
  fit gemma3-1b cuda-resident "$SCUDA" -model bench=$PG1 -backend cuda -ctx 4096 -quant int4 || fok=1
  fit phi3-mini cuda-resident "$SCUDA" -model bench=$PPHI -backend cuda -ctx 4096 || fok=1
  fit ttft-1.5B cuda-resident "$SCUDA" -model bench=$P15 -backend cuda -quant int4 -require-backend || fok=1
  fit M26-ctx2048 cuda-resident "$SCUDA" -model bench=$PM26 -backend cuda -ctx 2048 -moe-cache-experts || fok=1
else
  fit cpu-0.5B cpu "$SCPU" -model bench=$P05 -backend cpu -ctx 4096 -quant int4 || fok=1
  fit cpu-1.5B cpu "$SCPU" -model bench=$P15 -backend cpu -ctx 4096 -quant int4 || fok=1
  fit cpu-7B cpu "$SCPU" -model bench=$P7 -backend cpu -ctx 4096 -quant int4 || fok=1
  fit cpu-phi3-mini cpu "$SCPU" -model bench=$PPHI -backend cpu -ctx 4096 || fok=1
  fit cpu-G20 cpu "$SCPU" -model bench=$PG20 -backend cpu -ctx 4096 || fok=1
fi
[ $fok = 0 ] || { echo "== a fit check refused or resolved the wrong path: nothing timed; exit 75 $(ts)"; exit 75; }
echo "== fit checks done $(ts)"
[ "${FIT_ONLY:-}" = 1 ] && exit 0

# --- the cells ------------------------------------------------------------------------------------------------------------
export GOINFER_SERVE_CUDA=$SCUDA GOINFER_SERVE_CPU=$SCPU GOINFER_SERVE=$SCUDA
export OLLAMA_BIN=$HOME/ollama-0325/bin/ollama OLLAMA_MODELS=$HOME/ollama-0325/models
unset LD_LIBRARY_PATH   # 09-25: Ollama's lib dir made llama-server load the wrong libllama
export LLAMACPP_BIN=$HOME/mycode/peers/llama.cpp/build/bin/llama-server
export BENCH_RUNS=3 BENCH_QUANT_OVERRIDE="M35=int4mix,phi3-mini=default,G20=default"
export BENCH_SWAP_VOID_STOP=3 BENCH_SWAP_KILL_MB=1024
rc_all=0
SETTLE_LOAD=0.8 SETTLE_MAX_S=600
settle() { # wait for the 1-minute load average to fall under SETTLE_LOAD, logging each reading; 1 if it never does
  local t0 la; t0=$(date +%s)
  while :; do
    la=$(cut -d' ' -f1 /proc/loadavg)
    awk -v l="$la" -v c="$SETTLE_LOAD" 'BEGIN{exit !(l<c)}' && { echo "=== settle: load $la < $SETTLE_LOAD after $(( $(date +%s) - t0 ))s"; return 0; }
    [ $(( $(date +%s) - t0 )) -ge $SETTLE_MAX_S ] && { echo "=== settle: load $la still >= $SETTLE_LOAD after ${SETTLE_MAX_S}s"; return 1; }
    echo "=== settle: load $la >= $SETTLE_LOAD, waiting ($(( $(date +%s) - t0 ))/${SETTLE_MAX_S}s)"
    sleep 20
  done
}
step() { # step <name> <results.json> <command...>: one harness invocation; a refusal or an empty results file is a failure
  local name=$1 res=$2; shift 2
  if [ -n "$STEPS" ] && [[ ",$STEPS," != *",$name,"* ]]; then echo "=== skip $name (not in $STEPS)"; return 0; fi
  [ -e "$OUT/$name.log" ] && mv "$OUT/$name.log" "$OUT/$name.attempt-$(date -r "$OUT/$name.log" +%Y%m%dT%H%M%S).log"
  echo "=== evict: $(python3 "$HERE/evict_cache.py" "$HOME/models" "$OLLAMA_MODELS" 2>&1) $(state)"
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
B=scripts/bench_peer.py
E3=goinfer,ollama,llamacpp
if [ "$PART" = cuda ]; then
  step a-dense "$OUT/a-dense.json" env BENCH_ENGINES=$E3 BENCH_MODELS=0.5B,1.5B,7B BENCH_BACKENDS=cuda BENCH_DEPTHS=2048,3900 \
    python3 -u $B "$OUT/a-dense.json"
  step a-depth8000 "$OUT/a-depth8000.json" env BENCH_ENGINES=$E3 BENCH_CTX=8192 BENCH_MODELS=0.5B,1.5B,7B BENCH_BACKENDS=cuda \
    BENCH_DEPTHS=8000 python3 -u $B "$OUT/a-depth8000.json"
  step b-controls "$OUT/b-controls.json" env BENCH_ENGINES=$E3 BENCH_MODELS=gemma3-1b,phi3-mini BENCH_BACKENDS=cuda BENCH_DEPTHS=3900 \
    python3 -u $B "$OUT/b-controls.json"
  step f-sampled "$OUT/f-sampled.json" env BENCH_ENGINES=$E3 BENCH_MODELS=0.5B,gemma3-1b,phi3-mini BENCH_BACKENDS=cuda BENCH_DEPTHS=none \
    BENCH_CONFIGS=temp0.8_topk40_topp0.9,temp1.0_notrunc,temp0.8_topp0.95 python3 -u $B "$OUT/f-sampled.json"
  step d-prefill "$OUT/d-prefill.json" python3 -u scripts/bench_peer_prefill.py "$OUT/d-prefill.json" --models 1.5B --depths 512,3900 \
    --n 6 --backend cuda
  step c-26b "$OUT/c-26b.json" env BENCH_ENGINES=$E3 BENCH_CTX=2048 BENCH_MODELS=M26 BENCH_BACKENDS=cuda BENCH_DEPTHS=none \
    python3 -u $B "$OUT/c-26b.json"
else
  step e-cpu "$OUT/e-cpu.json" env BENCH_ENGINES=$E3 BENCH_MODELS=0.5B,1.5B,7B BENCH_BACKENDS=cpu BENCH_DEPTHS=none \
    python3 -u $B "$OUT/e-cpu.json"
  step v-cpu "$OUT/v-cpu.json" env BENCH_ENGINES=goinfer,ollama BENCH_MODELS=phi3-mini,G20 BENCH_BACKENDS=cpu BENCH_DEPTHS=none \
    python3 -u $B "$OUT/v-cpu.json"
  # v2 alone, both arms in a fresh file, run only when named: v-cpu.json already holds goinfer's night arm, which the
  # harness's resume would keep, pairing it with a later Ollama arm across sessions.
  if [ -n "$STEPS" ]; then
    step v2-g20 "$OUT/v2-g20.json" env BENCH_ENGINES=goinfer,ollama BENCH_MODELS=G20 BENCH_BACKENDS=cpu BENCH_DEPTHS=none \
      python3 -u $B "$OUT/v2-g20.json"
  fi
fi
echo "finished  $(ts) rc=$rc_all $(state)" | tee -a "$OUT/provenance.txt"
exit $rc_all

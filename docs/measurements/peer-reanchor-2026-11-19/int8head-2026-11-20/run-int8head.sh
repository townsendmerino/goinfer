#!/usr/bin/env bash
# The int8-head re-measure of the re-anchor cells #1179 moves (README.md beside this file), on main at #1179's merge.
#   bash run-int8head.sh build <rev>      # pinned worktree + binaries at <rev>, writes bin/SHA256SUMS (untimed; run under the lock, shared)
#   bash run-int8head.sh cuda|cpu [steps]  # the cells, as run-nobara.sh runs them: fit checks first (exit 75 on a refusal), the page
#                                          # cache evicted and the load settled before each step, one model per step
set -uo pipefail
PART=${1:?usage: run-int8head.sh build <rev> | cuda|cpu [step,step...]}
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
W=$HOME/goinfer-bench/peer-reanchor-int8head
if [ "$PART" = build ]; then
  REV=${2:?rev}; mkdir -p $W/bin
  [ -d $W/wt ] || git -C $HOME/mycode/reanchor-wt worktree add --detach $W/wt $REV || exit 1
  [ "$(git -C $W/wt rev-parse HEAD)" = "$(git -C $W/wt rev-parse $REV)" ] || { echo "wt is not at $REV"; exit 1; }
  (cd $W/wt && sh scripts/make_work.sh && cd backend/cuda && CGO_ENABLED=0 go build -tags cuda -o $W/bin/serve-cuda ./cmd/serve) || exit 1
  (cd $W/wt && CGO_ENABLED=0 go build -o $W/bin/serve-cpu ./cmd/serve) || exit 1
  git -C $W/wt rev-parse --short=9 HEAD > $W/bin/REV; (cd $W/bin && sha256sum serve-cuda serve-cpu > SHA256SUMS); cat $W/bin/REV $W/bin/SHA256SUMS; exit 0
fi
case $PART in cuda|cpu) ;; *) echo "unknown part $PART"; exit 2;; esac
STEPS=${2:-}
REV=$(cat $W/bin/REV)
DRIVER=595.112.07
H=$W/wt
SCUDA=$W/bin/serve-cuda
SCPU=$W/bin/serve-cpu
OUT=$W/results/$PART
L=$OUT/fit
mkdir -p "$OUT" "$L"
ts() { date '+%F %T %Z'; }
state() { echo "load $(cut -d' ' -f1-3 /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB swap $(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo) MiB"; }

# --- pins -------------------------------------------------------------------------------------------------------------
[ "$(git -C "$H" rev-parse --short=9 HEAD)" = "$REV" ] || { echo "FATAL: $H is not at $REV"; exit 2; }
[ -z "$(git -C "$H" status --porcelain --untracked-files=no)" ] || { echo "FATAL: $H has tracked changes"; exit 2; }
(cd $W/bin && sha256sum -c --quiet SHA256SUMS) || { echo "FATAL: the binaries are not the pinned build"; exit 2; }
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
  cat $W/bin/SHA256SUMS
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
  grep -q 'head table: int8' "$log" || { echo "FIT WRONG HEAD $name: $(grep -m1 'head table:' "$log")"; return 1; }
  echo "fit $name: $dp | $(grep -m1 ' fit:' "$log" | sed 's/^ *//') | $(grep -m1 'KV plan:' "$log" | sed 's/^ *//')"
}
echo "== fit checks $(ts) $(state)"
fok=0
if [ "$PART" = cuda ]; then
  for c in 4096 8192; do
    fit 1.5B-ctx$c cuda-resident "$SCUDA" -model bench=$P15 -backend cuda -ctx $c -quant int4 || fok=1
    fit 7B-ctx$c cuda-resident "$SCUDA" -model bench=$P7 -backend cuda -ctx $c -quant int4 || fok=1
  done
  fit ttft-1.5B cuda-resident "$SCUDA" -model bench=$P15 -backend cuda -quant int4 -require-backend || fok=1
else
  fit cpu-1.5B cpu "$SCPU" -model bench=$P15 -backend cpu -ctx 4096 -quant int4 || fok=1
  fit cpu-7B cpu "$SCPU" -model bench=$P7 -backend cpu -ctx 4096 -quant int4 || fok=1
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
  echo "=== evict: $(python3 "$HERE/../evict_cache.py" "$HOME/models" "$OLLAMA_MODELS" 2>&1) $(state)"
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
  for m in 1.5B 7B; do
    step a-$m "$OUT/a-$m.json" env BENCH_ENGINES=$E3 BENCH_MODELS=$m BENCH_BACKENDS=cuda BENCH_DEPTHS=2048,3900 python3 -u $B "$OUT/a-$m.json"
    step a8000-$m "$OUT/a8000-$m.json" env BENCH_ENGINES=$E3 BENCH_CTX=8192 BENCH_MODELS=$m BENCH_BACKENDS=cuda BENCH_DEPTHS=8000 \
      python3 -u $B "$OUT/a8000-$m.json"
  done
  step d-prefill "$OUT/d-prefill.json" python3 -u scripts/bench_peer_prefill.py "$OUT/d-prefill.json" --models 1.5B --depths 512,3900 \
    --n 6 --backend cuda
else
  for m in 1.5B 7B; do
    step e-$m "$OUT/e-$m.json" env BENCH_ENGINES=$E3 BENCH_MODELS=$m BENCH_BACKENDS=cpu BENCH_DEPTHS=none python3 -u $B "$OUT/e-$m.json"
  done
fi
echo "finished  $(ts) rc=$rc_all $(state)" | tee -a "$OUT/provenance.txt"
exit $rc_all

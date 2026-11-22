#!/usr/bin/env bash
# The x86 CPU peer cells re-measured at v0.25.0, registered in README.md beside this file before any data.
#   bash run.sh              # cells e (0.5B) and v (phi3-mini, gpt-oss 20B), by day, under the exclusive timing lock
#   FIT_ONLY=1 bash run.sh   # the fit checks alone, no timing
#   bash run.sh <step,step>  # only the named steps (e-cpu, v-cpu)
# ../peer-reanchor-2026-11-19/run-nobara.sh's cpu part with three changes: the pin is the v0.25.0 release asset
# goinfer-serve-linux-amd64 (sha256 from the release's checksums.txt) and bench_peer.py runs from a clean detached
# worktree at the tag; cell e holds the 0.5B only; results go under this campaign's own directory. The settle, the
# page-cache eviction, the fit checks, the harness settings and the grader are that script's, unchanged.
set -uo pipefail
STEPS=${1:-}
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
PREV=$HERE/../peer-reanchor-2026-11-19
REV=d0da204c
TAG=v0.25.0
W=$HOME/goinfer-bench/peer-cpu-v0.25.0-2026-11-22
H=$W/wt
SCPU=$W/bin/serve-v0.25.0-linux-amd64
SHA_CPU=e93cdf5928663b44da741bc44ea50d9b03b52d272abac2b2124daf36b46c53f9
OUT=$W/results/cpu
L=$OUT/fit
mkdir -p "$OUT" "$L"
ts() { date '+%F %T %Z'; }
state() { echo "load $(cut -d' ' -f1-3 /proc/loadavg) MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB swap $(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo) MiB"; }

# --- pins -------------------------------------------------------------------------------------------------------------
[ "$(git -C "$H" rev-parse --short=8 HEAD)" = "$REV" ] || { echo "FATAL: $H is not at $REV"; exit 2; }
[ "$(git -C "$H" rev-parse "$TAG^{commit}" | cut -c1-8)" = "$REV" ] || { echo "FATAL: $TAG is not $REV"; exit 2; }
[ -z "$(git -C "$H" status --porcelain --untracked-files=no)" ] || { echo "FATAL: $H has tracked changes"; exit 2; }
echo "$SHA_CPU  $SCPU" | sha256sum -c --quiet || { echo "FATAL: $SCPU is not the release asset"; exit 2; }
"$SCPU" -version 2>&1 | grep -q "v0.25.0 (d0da204c" || { echo "FATAL: $SCPU does not report v0.25.0 (d0da204c)"; exit 2; }
M=$HOME/models
P05=$M/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf PPHI=$M/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf PG20=$M/gpt-oss-20b-MXFP4.gguf
for p in $P05 $PPHI $PG20; do
  r=$(readlink -f "$p"); case "$r" in /srv/models*|/Volumes/*) echo "FATAL: $p resolves to the archive ($r)"; exit 2;; esac
  [ -e "$r" ] || { echo "FATAL: $p is missing"; exit 2; }
done
{ echo "part      cpu"; echo "started   $(ts)"; echo "rev       $TAG = $REV ($H, clean)"
  echo "serve     $SCPU sha256 $SHA_CPU ($("$SCPU" -version 2>&1 | head -1))"
  echo "host      $(hostname) $(uname -sr)"; echo "cpu       $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"
  echo "aikit     $(awk '/townsendmerino\/aikit /{print $2}' "$H/go.mod")"
  echo "ollama    $("$HOME/ollama-0325/bin/ollama" --version 2>&1 | tail -1)"
  echo "llama     $(LD_LIBRARY_PATH= "$HOME/mycode/peers/llama.cpp/build/bin/llama-server" --version 2>&1 | head -1)"
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
fit cpu-0.5B cpu "$SCPU" -model bench=$P05 -backend cpu -ctx 4096 -quant int4 || fok=1
fit cpu-phi3-mini cpu "$SCPU" -model bench=$PPHI -backend cpu -ctx 4096 || fok=1
fit cpu-G20 cpu "$SCPU" -model bench=$PG20 -backend cpu -ctx 4096 || fok=1
[ $fok = 0 ] || { echo "== a fit check refused or resolved the wrong path: nothing timed; exit 75 $(ts)"; exit 75; }
echo "== fit checks done $(ts)"
[ "${FIT_ONLY:-}" = 1 ] && exit 0

# --- the cells ------------------------------------------------------------------------------------------------------------
export GOINFER_SERVE_CPU=$SCPU GOINFER_SERVE=$SCPU
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
  echo "=== evict: $(python3 "$PREV/evict_cache.py" "$HOME/models" "$OLLAMA_MODELS" 2>&1) $(state)"
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
step e-cpu "$OUT/e-cpu.json" env BENCH_ENGINES=goinfer,ollama,llamacpp BENCH_MODELS=0.5B BENCH_BACKENDS=cpu BENCH_DEPTHS=none \
  python3 -u $B "$OUT/e-cpu.json"
step v-cpu "$OUT/v-cpu.json" env BENCH_ENGINES=goinfer,ollama BENCH_MODELS=phi3-mini,G20 BENCH_BACKENDS=cpu BENCH_DEPTHS=none \
  python3 -u $B "$OUT/v-cpu.json"
echo "finished  $(ts) rc=$rc_all $(state)" | tee -a "$OUT/provenance.txt"
exit $rc_all

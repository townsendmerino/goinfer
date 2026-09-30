#!/bin/bash
# Phi-3-mini on the MacBook: goinfer's CPU fallback vs Ollama on Metal. Pre-registered in
# docs/tasks/task-actquant-pergroup-2026-09.md, "Measure first", Mac arm (2026-09-29), before this ran.
#
# Three interleaved runs of bench_peer.py, phase A only (depth 128, greedy, 3 runs), one model, context pinned to 2048 on
# BOTH engines (q4k does not fit a 16 GB Mac at the default 4096: 9.2 GB needed, see the pre-registration):
#   a1  BENCH_BACKENDS=metal, goinfer with NO -quant flag (what a Mac user gets: int8int8 per-32, running on the CPU)  vs Ollama Metal
#   a2  BENCH_BACKENDS=metal, goinfer -quant q4k (the CPU fallback at nobara's default quant)                          vs Ollama Metal
#   b   BENCH_BACKENDS=cpu,   goinfer -quant q4k                                                                       vs Ollama CPU-only
# Ollama's tag is p3m-local (an alias of this Mac's p3m; 195/195 tensors identical to the GGUF, checked 2026-09-29).
# Binaries are pre-built at cbc6147e in $B. Each probe logs the decode path the load actually resolved to.
# Nothing here needs a Claude session; every result lands in $B (durable), the harness's own stamps included.
set -u
cd /Users/francistownsend-merino/tmcode/goinfer || exit 1
REV=cbc6147e
B=$HOME/goinfer-bench/phi3-metal-2026-09-29
MODEL=$HOME/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf
export OLLAMA_BIN=/opt/homebrew/bin/ollama OLLAMA_MODELS=$HOME/.ollama/models
export GOINFER_SERVE_METAL=$B/serve-metal-$REV GOINFER_SERVE_CPU=$B/serve-cpu-$REV
export BENCH_RUNS=3 BENCH_DEPTHS=none BENCH_MODELS=phi3-mini BENCH_CTX=2048 BENCH_ENGINES=goinfer,ollama
# a1 and a2 time the CPU fallback of a Metal request ON PURPOSE; the harness would otherwise void a GPU cell on the CPU. Its decode path is recorded.
export BENCH_ALLOW_CPU_FALLBACK=1

ts() { date '+%H:%M:%S'; }
echo "$(ts) start; tree $(git rev-parse --short HEAD) $(git status --porcelain scripts/bench_peer.py | head -1)"
test -f "$MODEL" || { echo "missing $MODEL"; exit 1; }
test -x "$GOINFER_SERVE_METAL" && test -x "$GOINFER_SERVE_CPU" || { echo "missing binaries in $B"; exit 1; }
ollama list | grep -q '^p3m-local' || { echo "missing ollama tag p3m-local"; exit 1; }

# probe <label> <binary> <backend> [extra serve flags...]: load once, ask one greedy question, keep the log.
probe() {
  local label=$1 bin=$2 be=$3; shift 3
  "$bin" -model "bench=$MODEL" -backend "$be" -ctx 2048 -addr 127.0.0.1:8198 "$@" > "$B/probe-$label.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8198/v1/models >/dev/null 2>&1 && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
  curl -s -m 90 http://127.0.0.1:8198/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"bench","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":16,"temperature":0}' \
    > "$B/probe-$label.reply.json" 2>&1
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  echo "$(ts) probe $label: $(grep -m1 'decode path' "$B/probe-$label.log" | cut -c1-200)"
  grep -m1 'NOT loaded\|does not fit' "$B/probe-$label.log" | cut -c1-200
}
probe a1-metal-default  "$GOINFER_SERVE_METAL" metal
probe a2-metal-q4k      "$GOINFER_SERVE_METAL" metal -quant q4k
probe b-cpu-q4k         "$GOINFER_SERVE_CPU"   cpu   -quant q4k

run() { # <name> <backend> <quant-override>
  echo "$(ts) == $1 start"
  BENCH_BACKENDS=$2 BENCH_QUANT_OVERRIDE="phi3-mini=$3" python3 -u scripts/bench_peer.py "$B/$1.json"
  echo "$(ts) == $1 exit=$?"
}
run a1-metal-default metal default
run a2-metal-q4k     metal q4k
run b-cpu-q4k        cpu   q4k
echo "$(ts) == DONE"

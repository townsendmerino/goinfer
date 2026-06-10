#!/usr/bin/env bash
# bench_compare.sh — reproduce goinfer's column of docs/benchmarks.md end-to-end,
# and print (NOT run) the peer commands so you can fill their columns on the SAME
# machine. Peers' install is yours; this script never drives them.
#
# It measures goinfer's side: steady-state decode tok/s (BenchmarkDecode), a timed
# prefill (BenchmarkPrefillLong), cold-start wall clock to first token, and resident
# memory (phys_footprint on macOS, max-RSS on Linux). Every number it prints carries
# the goinfer commit + date + machine so it can be pasted into the table with
# provenance intact.
#
# Usage:
#   GINFER_PREQUANT_GGUF=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
#     scripts/bench_compare.sh
#
# Env:
#   GINFER_PREQUANT_GGUF  model checkpoint for the Go benchmarks (a .giw or .gguf).
#                         Defaults to the in-repo 0.5B testdata GGUF if present.
#   GINFER_PREFILL_LEN    prefill length for the TTFT/prefill bench (default 1024).
#   GINFER_GPU            set to 1 to also run the -tags gpu residency BenchmarkDecode
#                         (needs a webgpu-capable build + an int8/int4 .giw).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
DATE="$(date -u +%Y-%m-%d)"
OS="$(uname -s)"
CPU="unknown"
case "$OS" in
  Darwin) CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo Apple)" ;;
  Linux)  CPU="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | sed 's/.*: //' || echo Linux)" ;;
esac

MODEL="${GINFER_PREQUANT_GGUF:-testdata/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf}"
PREFILL_LEN="${GINFER_PREFILL_LEN:-1024}"

echo "═══════════════════════════════════════════════════════════════════════"
echo " goinfer bench_compare — provenance for docs/benchmarks.md"
echo "   goinfer commit : $COMMIT"
echo "   date (UTC)     : $DATE"
echo "   machine        : $OS · $CPU"
echo "   model          : $MODEL"
echo "   NOTE: plug in, run 2–3×, take the median; record thermal state."
echo "═══════════════════════════════════════════════════════════════════════"

if [ ! -e "$MODEL" ]; then
  echo "!! model not found: $MODEL — set GINFER_PREQUANT_GGUF. Skipping Go benches."
else
  echo
  echo "── goinfer: steady-state decode tok/s (BenchmarkDecode) ──────────────"
  GINFER_PREQUANT_GGUF="$MODEL" go test ./decoder -run '^$' \
    -bench '^BenchmarkDecode$' -benchtime 30x 2>/dev/null | grep -E 'Benchmark|tok/s' || \
    echo "  (BenchmarkDecode unavailable — check the model loads at int8int8)"

  echo
  echo "── goinfer: prefill tok/s @ ${PREFILL_LEN} tokens (BenchmarkPrefillLong) ──"
  GINFER_PREQUANT_GGUF="$MODEL" GINFER_PREFILL_LEN="$PREFILL_LEN" go test ./decoder -run '^$' \
    -bench '^BenchmarkPrefillLong$' -benchtime 3x 2>/dev/null | grep -E 'Benchmark|tok/s|canBatchN' || \
    echo "  (BenchmarkPrefillLong unavailable)"

  if [ "${GINFER_GPU:-0}" = "1" ]; then
    echo
    echo "── goinfer: GPU residency decode (-tags gpu BenchmarkDecode) ─────────"
    GINFER_PREQUANT_GGUF="$MODEL" go test -tags gpu ./decoder -run '^$' \
      -bench '^BenchmarkDecode$' -benchtime 30x 2>/dev/null | grep -E 'Benchmark|tok/s' || \
      echo "  (GPU bench unavailable — needs a webgpu build + resident-eligible .giw)"
  fi

  echo
  echo "── goinfer: cold start → first token (wall clock) + resident memory ──"
  # Build the demo once, then time a one-token generation from cold (process start
  # to first token) and capture peak resident memory the OS-appropriate way.
  go build -o /tmp/goinfer_chat ./demo/chat 2>/dev/null
  if [ -x /tmp/goinfer_chat ]; then
    if [ "$OS" = "Darwin" ]; then
      # macOS: /usr/bin/time -l reports "maximum resident set size" (phys_footprint-class)
      /usr/bin/time -l /tmp/goinfer_chat --model "$MODEL" --max-tokens 1 --prompt "hi" 2>&1 \
        | grep -iE 'real|maximum resident' || echo "  (timing unavailable)"
    else
      # Linux: GNU time -v reports "Maximum resident set size (kbytes)"
      /usr/bin/time -v /tmp/goinfer_chat --model "$MODEL" --max-tokens 1 --prompt "hi" 2>&1 \
        | grep -iE 'Elapsed \(wall|Maximum resident' || echo "  (install GNU time, or use the demo's own timing)"
    fi
  else
    echo "  (demo/chat build failed — adjust flags for your demo entrypoint)"
  fi
fi

cat <<'PEERS'

═══════════════════════════════════════════════════════════════════════
 PEER COMMANDS — run these YOURSELF on the SAME machine, same checkpoint,
 same quant, greedy/fixed seed. Record each peer's version + the date.
 (This script does not run them: their install is yours.)
═══════════════════════════════════════════════════════════════════════

# Ollama (decode tok/s; ensure `ollama ps` shows 100% GPU for the GPU row):
#   ollama --version
#   ollama run qwen2.5-coder:1.5b-instruct-q8_0 --verbose "<<your fixed prompt>>"
#     → read "eval rate (tokens/s)". Match the quant to goinfer's (q8_0 ≈ int8).

# llama.cpp (decode + prefill, same GGUF goinfer loads):
#   llama-cli --version
#   llama-bench -m qwen2.5-coder-1.5b-instruct-q8_0.gguf -p 512 -n 128 -ngl 99
#     → "pp" = prefill tok/s, "tg" = decode tok/s. -ngl 99 = all layers on GPU.

# vLLM (server throughput; not single-stream comparable — note that in the table):
#   python -c "import vllm; print(vllm.__version__)"
#   vllm serve <model> ; then benchmark with vllm's bench_serving.

# IMPORTANT: a number only enters docs/benchmarks.md if peer + goinfer ran the
# SAME model checkpoint at the SAME quant on the SAME machine. Mismatched runs
# (e.g. a different param count or quant) are how you get a wrong comparison —
# see docs/gpu-assessment.md, which caught exactly that (a 1.8B q4 mistaken for
# the 1.5B). When in doubt, REMEASURE; never paste a number you can't trace.
PEERS

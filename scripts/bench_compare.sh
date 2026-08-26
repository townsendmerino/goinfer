#!/usr/bin/env bash
# bench_compare.sh — goinfer's OWN numbers only. NOT a peer comparison.
#
# ⚠ DO NOT USE THIS FOR PEER FIGURES. It measures goinfer with IN-PROCESS Go benchmarks
# (BenchmarkDecode et al) and does not drive any peer. Pasting its output beside a peer's
# server-measured number compares a kernel throughput against an end-to-end throughput — exactly
# how the retired "0.5B 476 vs 268 / 1.78x" claim was produced (docs/benchmarks.md B2, retired
# 2026-08-09). For a defensible comparison use scripts/bench_peer.py, which drives BOTH sides over
# HTTP, interleaved, with a restart between cells and sampling recorded on each.
#
# It measures goinfer's side: steady-state decode tok/s (BenchmarkDecode), a timed
# prefill (BenchmarkPrefillLong), cold-start wall clock to first token, and resident
# memory (phys_footprint on macOS, max-RSS on Linux). Every number it prints carries
# the goinfer commit + date + machine so it can be pasted into the table with
# provenance intact.
#
# Usage:
#   GOINFER_PREQUANT_GGUF=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
#     scripts/bench_compare.sh
#
# Env:
#   GOINFER_PREQUANT_GGUF  model checkpoint for the Go benchmarks (a .giw or .gguf).
#                         Defaults to the in-repo 0.5B testdata GGUF if present.
#   GOINFER_PREFILL_LEN    prefill length for the TTFT/prefill bench (default 1024).
#   GOINFER_GPU            set to 1 to also run the -tags gpu residency BenchmarkDecode
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

MODEL="${GOINFER_PREQUANT_GGUF:-testdata/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf}"
PREFILL_LEN="${GOINFER_PREFILL_LEN:-1024}"

echo "═══════════════════════════════════════════════════════════════════════"
echo " goinfer bench_compare — provenance for docs/benchmarks.md"
echo "   goinfer commit : $COMMIT"
echo "   date (UTC)     : $DATE"
echo "   machine        : $OS · $CPU"
echo "   model          : $MODEL"
echo "   NOTE: plug in, run 2–3×, take the median; record thermal state."
echo "═══════════════════════════════════════════════════════════════════════"

# A timed run must read its checkpoint from the local bench set, never from the archive. Both
# roots are checked because neither catches the other: /Volumes/ does not exist on Linux, and
# /srv/models is a LOCAL mount on the box that measures every CUDA row, so "benchmark from local
# disk" reads as permission for it. Reading the 5400 rpm SMR archive does not error -- it returns
# a plausible, wrong number. realpath, so a symlink from ~/models into the archive is caught too.
# Authority: docs/benchmarks.md, "Model storage".
REAL_MODEL="$(realpath "$MODEL" 2>/dev/null || echo "$MODEL")"
case "$REAL_MODEL" in
  /srv/models/*|/Volumes/*)
    echo "!! REFUSED: $MODEL resolves to $REAL_MODEL, which is on the ARCHIVE."
    echo "!! The archive is a 5400 rpm SMR disk and is a bench surface for neither machine."
    echo "!! A run off it measures that disk, not the engine, and does NOT error."
    echo "!! Copy to the local bench set first (models-pull <name>) and re-run."
    exit 2
    ;;
esac

if [ ! -e "$MODEL" ]; then
  echo "!! model not found: $MODEL — set GOINFER_PREQUANT_GGUF. Skipping Go benches."
else
  echo
  echo "── goinfer: steady-state decode tok/s (BenchmarkDecode) ──────────────"
  GOINFER_PREQUANT_GGUF="$MODEL" go test ./decoder -run '^$' \
    -bench '^BenchmarkDecode$' -benchtime 30x 2>/dev/null | grep -E 'Benchmark|tok/s' || \
    echo "  (BenchmarkDecode unavailable — check the model loads at int8int8)"

  echo
  echo "── goinfer: prefill tok/s @ ${PREFILL_LEN} tokens (BenchmarkPrefillLong) ──"
  GOINFER_PREQUANT_GGUF="$MODEL" GOINFER_PREFILL_LEN="$PREFILL_LEN" go test ./decoder -run '^$' \
    -bench '^BenchmarkPrefillLong$' -benchtime 3x 2>/dev/null | grep -E 'Benchmark|tok/s|canBatchN' || \
    echo "  (BenchmarkPrefillLong unavailable)"

  if [ "${GOINFER_GPU:-0}" = "1" ]; then
    echo
    echo "── goinfer: GPU residency decode (-tags gpu BenchmarkDecode) ─────────"
    GOINFER_PREQUANT_GGUF="$MODEL" go test -tags gpu ./decoder -run '^$' \
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
 PEER COMMANDS — reference only. Numbers you obtain this way are NOT comparable
 with the goinfer figures above (those are in-process benchmarks, these are
 end-to-end server measurements). For a publishable comparison run
 scripts/bench_peer.py instead, which measures both sides identically.
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

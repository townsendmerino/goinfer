#!/usr/bin/env bash
# L1 (pre-registered cc1769c7): int4 f32 scales vs f16-rounded scales, per-row activations, 12 families x 2 prompts.
set -u
D=/home/francis/goinfer-bench/cpu-decode-2026-09-27; M=$HOME/models
for spec in \
  phi3-mini=$M/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  qwen2.5-7b=$M/qwen2.5-7b-instruct-q4_k_m.gguf \
  qwen2.5-coder-1.5b=$M/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  qwen3-1.7b=$M/qwen3-1.7b-bf16 \
  qwen3.5-0.8b=$M/qwen3.5-0.8b \
  gemma3-1b=$M/gemma3-1b-q4_k_m.gguf \
  llama3.2-1b=$M/llama-3.2-1b-instruct-q4_k_m.gguf \
  tinyllama-1.1b=$M/tinyllama-1.1b-chat \
  mistral-7b=$M/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
  granite-4.2-3b=$M/granite-4.2-3b \
  smollm3-3b=$M/smollm3-3b \
  olmo3-7b=$M/olmo3-7b-think ; do
  name=${spec%%=*}; path=${spec#*=}
  echo "=== $(date '+%F %T %Z') START $name"; t0=$(date +%s)
  timeout 5400 $D/f16sweep "$name" "$path" >> $D/f16sweep.jsonl 2>> $D/f16sweep-errors.log
  echo "=== $(date '+%F %T %Z') END $name rc=$? ($(( $(date +%s) - t0 ))s)"
done
echo "=== $(date '+%F %T %Z') ALL DONE"

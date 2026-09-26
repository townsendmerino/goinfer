#!/usr/bin/env bash
# Track B (pre-registered a4e6413b): int4 weight-quantizer candidates, per-32 activations ON, int4 only.
D=$HOME/goinfer-logs/actquant-trackb; M=$HOME/models
for scheme in fullrange mse; do
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
  echo "=== $(date -u +%FT%TZ) START $scheme $name"; t0=$(date +%s)
  INT4_SCHEME=$scheme ACT_GROUP=32 QUANTS=int4 timeout 5400 $D/actsweep "$name" "$path" >> $D/sweep-$scheme.jsonl 2>> $D/errors.log
  echo "=== $(date -u +%FT%TZ) END $scheme $name rc=$? ($(( $(date +%s) - t0 ))s)"
done; done
echo "=== $(date -u +%FT%TZ) ALL DONE"

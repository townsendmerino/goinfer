#!/usr/bin/env bash
# (Archived copy. The tool source is actsweep.go.txt here, renamed from actsweep.go so the Go toolchain ignores it.)
# H2 step 1: per family, goinfer int8 / int8int8 / int4 vs its own f32, per position, on two ~130-token
# chat prompts (actsweep.go). int8 vs int8int8 isolates per-row activation quantization.
D=$HOME/goinfer-logs/actquant-sweep; M=$HOME/models
until grep -q "DONE" $HOME/goinfer-logs/prompts-essay-v2/validate.log 2>/dev/null; do sleep 30; done
for spec in \
  phi3-mini=$M/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  qwen2.5-coder-1.5b=$M/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  qwen2.5-7b=$M/qwen2.5-7b-instruct-q4_k_m.gguf \
  qwen3-1.7b=$M/qwen3-1.7b-bf16 \
  qwen3.5-0.8b=$M/qwen3.5-0.8b \
  gemma3-1b=$M/gemma3-1b-q4_k_m.gguf \
  gemma4-E2B=$M/gemma-4-E2B-unq \
  llama3.2-1b=$M/llama-3.2-1b-instruct-q4_k_m.gguf \
  tinyllama-1.1b=$M/tinyllama-1.1b-chat \
  mistral-7b=$M/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
  granite-4.2-3b=$M/granite-4.2-3b \
  smollm3-3b=$M/smollm3-3b \
  lfm2.5-2.6b=$M/lfm25-2.6b \
  olmo3-7b=$M/olmo3-7b-think ; do
  name=${spec%%=*}; path=${spec#*=}
  echo "=== $(date -u +%FT%TZ) START $name"; t0=$(date +%s)
  timeout 3600 $D/actsweep "$name" "$path" >> $D/sweep.jsonl 2>> $D/errors.log
  echo "=== $(date -u +%FT%TZ) END $name rc=$? ($(( $(date +%s) - t0 ))s)"
done
echo "=== $(date -u +%FT%TZ) ALL DONE"

#!/usr/bin/env bash
# Phase 1a quality gate (registered 306534a7): arm R = a real --quant q4k load (AVX2 kernel, aikit working tree), beside A0.
D=$HOME/goinfer-logs/q4k-gate-1a; M=$HOME/models
cd $HOME/mycode/goinfer
for spec in \
  llama3.2-1b=$M/llama-3.2-1b-instruct-q4_k_m.gguf \
  gemma3-1b=$M/gemma3-1b-q4_k_m.gguf \
  qwen2.5-coder-1.5b=$M/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  phi3-mini=$M/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf \
  mistral-7b=$M/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
  qwen2.5-7b=$M/qwen2.5-7b-instruct-q4_k_m.gguf ; do
  name=${spec%%=*}; path=${spec#*=}
  echo "=== $(date -u +%FT%TZ) START $name"; t0=$(date +%s)
  ARMS=R,A0 TYPES_JSON=$D/types-$name.json timeout 5400 $D/q4kgate $name $path >> $D/sweep.jsonl 2>> $D/progress.log
  echo "=== $(date -u +%FT%TZ) END $name rc=$? ($(( $(date +%s)-t0 ))s)"
done
echo "=== $(date -u +%FT%TZ) ALL DONE"

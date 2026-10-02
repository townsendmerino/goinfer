#!/bin/bash
p='Write a Go function that reverses a string.'
echo "=== 1. plain chat, 0.5B gguf, default auto ($(date +%H:%M:%S))"
printf '%s\n' "$p" | ./goinfer-chat-r17-linux-amd64 --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf --direct-load --temp 0 --max 48 2>&1 | grep -v '^\s*$' | head -12
echo "=== 2. same, GPU hidden (CUDA_VISIBLE_DEVICES=-1) ($(date +%H:%M:%S))"
printf '%s\n' "$p" | CUDA_VISIBLE_DEVICES=-1 ./goinfer-chat-r17-linux-amd64 --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf --direct-load --temp 0 --max 48 2>&1 | grep -v '^\s*$' | head -12
echo "=== 3. model-included 0.5B (int8int8), default auto ($(date +%H:%M:%S))"
printf '%s\n' "$p" | ./r17-smoke-linux-amd64 --temp 0 --max 48 2>&1 | grep -v '^\s*$' | head -12
echo "=== done ($(date +%H:%M:%S))"

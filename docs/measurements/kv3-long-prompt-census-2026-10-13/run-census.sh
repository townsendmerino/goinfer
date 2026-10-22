#!/bin/bash
# the long-prompt census, one model after another (docs/measurements/kv3-long-prompt-census-2026-10-13.md)
S=${S:-$HOME/goinfer-bench/kv3-2026-10/bin/serve-cuda-new}; M=$HOME/models; O=${O:-$HOME/goinfer-bench/kv3-2026-10/census}; H=$(dirname "$(readlink -f "$0")")
mkdir -p "$O"
run() { label=$1; path=$2; shift 2; echo "== $(date +%T) $label"; timeout 1500 python3 -u "$H/census.py" "$S" "$path" "$label" "$O/$label.json" -- "$@" > "$O/$label.out" 2>&1; echo "   rc=$?"; }
run gemma-3-4b "$M/gemma-3-4b-it-Q4_K_M.gguf" --ctx 24576
run gpt-oss-20b "$M/gpt-oss-20b-MXFP4.gguf" --moe-cache-experts --ctx 24576
run mellum2.1 "$M/mellum2.1-thinking" --moe-cache-experts --ctx 24576
run gemma-4-e4b "$M/gemma-4-e4b-gguf/gemma-4-E4B_q4_0-it.gguf" --ctx 24576
run qwen3.6-35b-a3b "$M/qwen3.6-35b-a3b-int4.giw" --moe-cache-experts --ctx 24576
echo "== $(date +%T) done"

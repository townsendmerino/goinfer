#!/bin/bash
# census follow-up: no --ctx banners for the models that decline, and gpt-oss past the 12,160-key bound
S=${S:-$HOME/goinfer-bench/kv3-2026-10/bin/serve-cuda-new}; M=$HOME/models; O=${O:-$HOME/goinfer-bench/kv3-2026-10/census}; H=$(dirname "$(readlink -f "$0")")
run() { mode=$1; words=$2; label=$3; path=$4; shift 4; echo "== $(date +%T) $label $mode"; CENSUS_MODE=$mode CENSUS_WORDS=$words timeout 900 python3 -u "$H/census2.py" "$S" "$path" "$label" "$O/$label-$mode.json" -- "$@" > "$O/$label-$mode.out" 2>&1; echo "   rc=$?"; }
run banner 0 gemma-3-4b "$M/gemma-3-4b-it-Q4_K_M.gguf"
run banner 0 gemma-4-e4b "$M/gemma-4-e4b-gguf/gemma-4-E4B_q4_0-it.gguf"
run long 2150 gpt-oss-20b "$M/gpt-oss-20b-MXFP4.gguf" --moe-cache-experts --ctx 24576
echo "== $(date +%T) done"

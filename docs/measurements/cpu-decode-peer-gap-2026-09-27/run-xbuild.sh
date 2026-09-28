#!/usr/bin/env bash
# L1 build gate 2: CPU int4 logits, new build (aikit f16 storage) vs old build (main, f32 storage) with
# GOINFER_INT4_F16_SCALES=1. Byte-identical dumps required.
set -u
D=/home/francis/goinfer-bench/cpu-decode-2026-09-27
for m in qwen2.5-coder-0.5b-instruct-q4_k_m qwen2.5-coder-1.5b-instruct-q4_k_m qwen2.5-7b-instruct-q4_k_m; do
  p=$HOME/models/$m.gguf
  echo "=== $(date '+%T') $m old"
  GOINFER_INT4_F16_SCALES=1 GOINFER_NO_FIT_GUARD=1 $D/f16xbuild-old $p $D/xb-old-$m.bin || { echo "old FAILED"; continue; }
  echo "=== $(date '+%T') $m new"
  GOINFER_NO_FIT_GUARD=1 $D/f16xbuild-new $p $D/xb-new-$m.bin || { echo "new FAILED"; continue; }
  if cmp -s $D/xb-old-$m.bin $D/xb-new-$m.bin; then
    echo "=== $m: IDENTICAL ($(stat -c %s $D/xb-new-$m.bin) bytes, $(( $(stat -c %s $D/xb-new-$m.bin) / 4 )) logits)"
  else
    echo "=== $m: DIFFER"; cmp $D/xb-old-$m.bin $D/xb-new-$m.bin | head -2
  fi
done
echo "=== $(date '+%T') DONE"

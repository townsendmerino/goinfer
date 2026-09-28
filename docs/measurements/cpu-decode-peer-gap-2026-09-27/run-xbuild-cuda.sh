#!/usr/bin/env bash
# L1 build gate 3 (CUDA half): CUDA int4 logits, new build vs old build — byte-identical required.
set -u
D=/home/francis/goinfer-bench/cpu-decode-2026-09-27
for m in qwen2.5-coder-0.5b-instruct-q4_k_m qwen2.5-coder-1.5b-instruct-q4_k_m qwen2.5-7b-instruct-q4_k_m; do
  p=$HOME/models/$m.gguf
  echo "=== $(date '+%T') $m old"; $D/f16xbuild-cuda-old $p $D/xbc-old-$m.bin || { echo "old FAILED"; continue; }
  echo "=== $(date '+%T') $m new"; $D/f16xbuild-cuda-new $p $D/xbc-new-$m.bin || { echo "new FAILED"; continue; }
  if cmp -s $D/xbc-old-$m.bin $D/xbc-new-$m.bin; then echo "=== $m: IDENTICAL ($(( $(stat -c %s $D/xbc-new-$m.bin) / 4 )) logits)"; else echo "=== $m: DIFFER"; cmp $D/xbc-old-$m.bin $D/xbc-new-$m.bin | head -2; fi
done
echo "=== $(date '+%T') DONE"

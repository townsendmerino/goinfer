#!/bin/bash
# L1 arm64 fix, hard gate 3: CPU arm64 logits with the aikit fix (2fd6f59, local replace on goinfer e351fad4) vs the
# PR-head build 84ee8f49's dumps (same decoder; 84ee8f49 == old-with-f16-rounded-scales, proven earlier). cmp decides.
set -u
L=$HOME/goinfer-bench/l1-mac; M=$HOME/models; cd $L
run() { # tag model tokenizer
  echo "=== $(date +%H:%M:%S) cpu $1 fix ($(basename $2))"
  ./xb-cpu-fix "$2" "$3" cpu dump-cpu-$1-fix.bin 2>&1 | grep -v "fit is tight"
  echo "exit ${PIPESTATUS[0]}"
  if cmp -s dump-cpu-$1-fix.bin dump-cpu-$1-new.bin; then
    echo "=== cpu $1: fix vs 84ee8f49 IDENTICAL ($(stat -f %z dump-cpu-$1-fix.bin) bytes)"
  else
    echo "=== cpu $1: DIFFER $(cmp dump-cpu-$1-fix.bin dump-cpu-$1-new.bin | head -1)"
  fi
}
run 1.5b $M/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf $M/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
run 7b $M/qwen2.5-7b-instruct-q4_k_m.int4.cpu-arm64.giw $M/qwen2.5-7b-instruct-q4_k_m.gguf

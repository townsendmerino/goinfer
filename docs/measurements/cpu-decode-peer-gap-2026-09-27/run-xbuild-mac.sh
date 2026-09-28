#!/bin/bash
# L1 Mac identity (gates 2 and 3): old (origin/main 3cd62e6d, aikit v1.49.0) vs new (f16-scales 84ee8f49, aikit v1.50.0)
# full-logit dumps on Metal, WebGPU and arm64 CPU, then cmp. Old CPU runs with GOINFER_INT4_F16_SCALES=1.
set -u
L=$HOME/goinfer-bench/l1-mac; M=$HOME/models
ts() { date '+%H:%M:%S'; }
T15=$M/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf; T7=$M/qwen2.5-7b-instruct-q4_k_m.gguf
src() { # backend model -> source path
  case "$1:$2" in
    *:1.5b) echo "$T15";;
    metal:7b) echo "$M/qwen2.5-7b-instruct-q4_k_m.int4.metal.giw";;
    cpu:7b) echo "$M/qwen2.5-7b-instruct-q4_k_m.int4.cpu-arm64.giw";;
    webgpu:7b) echo "$T7";;
  esac; }
for be in metal webgpu cpu; do
  for mdl in 1.5b 7b; do
    s=$(src $be $mdl); tok=$T15; [ $mdl = 7b ] && tok=$T7
    for bld in old new; do
      env=""; [ $be = cpu ] && [ $bld = old ] && env="GOINFER_INT4_F16_SCALES=1"
      echo "=== $(ts) $be $mdl $bld ($(basename $s)) $env"
      env $env $L/xb-$be-$bld "$s" "$tok" $be $L/dump-$be-$mdl-$bld.bin 2>&1 | grep -vE "^\s*$" | tail -6
      echo "=== $(ts) $be $mdl $bld exit ${PIPESTATUS[0]}"
    done
    if [ -s $L/dump-$be-$mdl-old.bin ] && [ -s $L/dump-$be-$mdl-new.bin ]; then
      if cmp -s $L/dump-$be-$mdl-old.bin $L/dump-$be-$mdl-new.bin; then
        echo "=== $be $mdl: IDENTICAL ($(stat -f %z $L/dump-$be-$mdl-new.bin) bytes, $(( $(stat -f %z $L/dump-$be-$mdl-new.bin) / 4 )) logits)"
      else
        echo "=== $be $mdl: DIFFER — $(cmp $L/dump-$be-$mdl-old.bin $L/dump-$be-$mdl-new.bin | head -1); sizes $(stat -f %z $L/dump-$be-$mdl-old.bin) / $(stat -f %z $L/dump-$be-$mdl-new.bin)"
      fi
    else
      echo "=== $be $mdl: MISSING a dump"
    fi
    rm -f $L/dump-$be-$mdl-old.bin.keep
  done
done
echo "=== $(ts) DONE"

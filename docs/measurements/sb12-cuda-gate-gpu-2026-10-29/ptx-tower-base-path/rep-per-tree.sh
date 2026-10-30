#!/usr/bin/env bash
# Compile tower_base.cu N times in each tree with V12.9.86 (the gate's toolchain), output to temp; print md5 per build.
V=$HOME/.venv-vl/lib/python3.14/site-packages/nvidia
LIB=$V/cuda_nvrtc/lib; INC=$V/cuda_runtime/include
T=$(mktemp -d)
for tree in nobara-ci01-cp-wt nobara-ci01-fr-wt; do
  cd "$HOME/goinfer-bench/$tree/cuda" || exit 2
  for i in 1 2 3 4; do
    LD_LIBRARY_PATH=$LIB NVRTC_SO=$LIB/libnvrtc.so.12 python3 nvrtc_compile.py ./tower_base.cu $T/o.ptx compute_75 $INC >/dev/null 2>&1
    echo "$tree $i $(md5sum < $T/o.ptx | cut -c1-12) $(grep -c . $T/o.ptx)"
  done
done
echo "committed $(md5sum < $HOME/goinfer-bench/nobara-ci01-fr-wt/cuda/testdata/tower_base.ptx | cut -c1-12)"

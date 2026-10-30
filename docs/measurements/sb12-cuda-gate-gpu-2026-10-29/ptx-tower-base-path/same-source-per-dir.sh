#!/usr/bin/env bash
# Same tower_base.cu, compiled from directories whose names differ only; md5 of the PTX per directory.
V=$HOME/.venv-vl/lib/python3.14/site-packages/nvidia; LIB=$V/cuda_nvrtc/lib; INC=$V/cuda_runtime/include
SRC=$HOME/goinfer-bench/nobara-ci01-fr-wt/cuda; B=$(mktemp -d)
for d in a bb ccc nobara-ci01-cp-wt nobara-ci01-fr-wt gate-gpu-cuda-sb12-wt x/y/z; do
  mkdir -p $B/$d/cuda; cp $SRC/tower_base.cu $SRC/nvrtc_compile.py $B/$d/cuda/
  (cd $B/$d/cuda && for i in 1 2; do LD_LIBRARY_PATH=$LIB NVRTC_SO=$LIB/libnvrtc.so.12 python3 nvrtc_compile.py ./tower_base.cu o$i.ptx compute_75 $INC >/dev/null 2>&1; done
   echo "$(printf %-22s $d) len=$(printf %3d ${#PWD}) $(md5sum < o1.ptx | cut -c1-12) $(md5sum < o2.ptx | cut -c1-12)")
done

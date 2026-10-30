#!/usr/bin/env bash
# One directory, one source; only the length of an unrelated environment variable changes. md5 per length (two builds each).
V=$HOME/.venv-vl/lib/python3.14/site-packages/nvidia; LIB=$V/cuda_nvrtc/lib; INC=$V/cuda_runtime/include
cd $HOME/goinfer-bench/nobara-ci01-fr-wt/cuda || exit 2; O=$(mktemp -d)
for n in 0 1 2 3 4 5 6 7 8 16 24 32; do
  pad=$(head -c $n /dev/zero | tr '\0' x)
  for i in 1 2; do PADDING_VAR="$pad" LD_LIBRARY_PATH=$LIB NVRTC_SO=$LIB/libnvrtc.so.12 python3 nvrtc_compile.py ./tower_base.cu $O/o$i.ptx compute_75 $INC >/dev/null 2>&1; done
  echo "pad=$(printf %2d $n) $(md5sum < $O/o1.ptx | cut -c1-12) $(md5sum < $O/o2.ptx | cut -c1-12)"
done

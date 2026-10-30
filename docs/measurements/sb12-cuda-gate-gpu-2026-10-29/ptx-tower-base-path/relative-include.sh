#!/usr/bin/env bash
# tower_base.cu from seven directories with nvrtc_compile.py's own-directory include path made relative ("."): md5 per dir.
V=$HOME/.venv-vl/lib/python3.14/site-packages/nvidia; LIB=$V/cuda_nvrtc/lib; INC=$V/cuda_runtime/include
SRC=$HOME/goinfer-bench/nobara-ci01-fr-wt/cuda; B=$(mktemp -d)
sed 's#os.path.dirname(os.path.abspath(src_path))#(os.path.dirname(src_path) or ".")#' $SRC/nvrtc_compile.py > $B/nvrtc_rel.py
grep -c 'os.path.dirname(src_path) or' $B/nvrtc_rel.py
for d in a bb nobara-ci01-cp-wt nobara-ci01-fr-wt gate-gpu-cuda-sb12-wt x/y/z q/r/s/t/u; do
  mkdir -p $B/$d/cuda; cp $SRC/tower_base.cu $B/$d/cuda/; cp $B/nvrtc_rel.py $B/$d/cuda/nvrtc_compile.py
  (cd $B/$d/cuda && LD_LIBRARY_PATH=$LIB NVRTC_SO=$LIB/libnvrtc.so.12 python3 nvrtc_compile.py ./tower_base.cu o.ptx compute_75 $INC >/dev/null 2>&1; echo "$(printf %-22s $d) $(md5sum < o.ptx | cut -c1-12)")
done

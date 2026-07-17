#!/usr/bin/env bash
# build_ptx.sh — regenerate cuda/testdata/*.ptx from the .cu sources in this directory.
#
# WHY THIS EXISTS
# ---------------
# The cgo-free CUDA backend embeds PTX (go:embed, cuda/kernels.go) and hands it to the
# driver's JIT at run time. That keeps the RUNTIME toolkit-free — but the PTX itself is a
# build-time artifact, and an artifact you cannot regenerate is an artifact you cannot
# review, audit, or change. Every shipped .ptx must be reproducible from a .cu in this
# directory by running this script.
#
# WHY NVRTC AND NOT NVCC
# ----------------------
# nvcc drives a HOST compiler and includes host C++ headers, so it inherits the host
# toolchain's constraints. On this box (gcc 15) nvcc's `cicc` cannot parse libstdc++'s
# bf16 literal in <c++config.h> and dies, even with -allow-unsupported-compiler; CUDA
# 12.6 also refuses gcc > 13 outright. NVRTC compiles CUDA C++ -> PTX with NO host
# compiler and no host headers, so it sidesteps the host-toolchain coupling entirely.
# It only needs the CUDA headers for <cuda_fp16.h> (the f16 group scales).
#
# USAGE
#   ./build_ptx.sh              # rebuild all kernels
#   ./build_ptx.sh fused_qkv    # rebuild one
#
# Override discovery with NVRTC_LIB (dir containing libnvrtc.so.12 AND
# libnvrtc-builtins.so.*) and CUDA_INC (dir containing cuda_fp16.h).
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${ARCH:-compute_75}" # RTX 2070 SUPER (Turing). PTX is forward-compatible via the JIT.

find_first() { for c in "$@"; do [ -e "$c" ] && { echo "$c"; return; }; done; }

if [ -z "${NVRTC_LIB:-}" ]; then
	lib=$(find_first \
		/tmp/cuda_extract/cuda_nvrtc/targets/x86_64-linux/lib/libnvrtc.so.12 \
		"$HOME"/cuda-toolkit/targets/x86_64-linux/lib/libnvrtc.so.12 \
		"$HOME"/.venv*/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 \
		/usr/local/cuda/lib64/libnvrtc.so.12 \
		/usr/lib64/libnvrtc.so.12 || true)
	[ -n "${lib:-}" ] && NVRTC_LIB="$(dirname "$lib")"
fi
if [ -z "${CUDA_INC:-}" ]; then
	hdr=$(find_first \
		/tmp/cuda_extract/cuda_cudart/targets/x86_64-linux/include/cuda_fp16.h \
		"$HOME"/cuda-toolkit/targets/x86_64-linux/include/cuda_fp16.h \
		"$HOME"/.venv*/lib/python*/site-packages/nvidia/cuda_runtime/include/cuda_fp16.h \
		/usr/local/cuda/include/cuda_fp16.h \
		/usr/include/cuda_fp16.h || true)
	[ -n "${hdr:-}" ] && CUDA_INC="$(dirname "$hdr")"
fi
if [ -z "${NVRTC_LIB:-}" ] || [ -z "${CUDA_INC:-}" ]; then
	echo "build_ptx: could not locate NVRTC and/or the CUDA headers." >&2
	echo "  NVRTC_LIB=${NVRTC_LIB:-<not found>}  (needs libnvrtc.so.12 + libnvrtc-builtins.so.*)" >&2
	echo "  CUDA_INC=${CUDA_INC:-<not found>}    (needs cuda_fp16.h)" >&2
	echo "Install with:  pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12" >&2
	echo "then re-run with NVRTC_LIB=... CUDA_INC=... $0" >&2
	exit 1
fi
echo "build_ptx: NVRTC_LIB=$NVRTC_LIB"
echo "build_ptx: CUDA_INC=$CUDA_INC"

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=($(ls ./*.cu | xargs -n1 basename | sed 's/\.cu$//'))

mkdir -p testdata
for k in "${targets[@]}"; do
	src="./${k%.cu}.cu"
	[ -f "$src" ] || { echo "build_ptx: no such kernel source: $src" >&2; exit 1; }
	# megakernel.cu is a spike artifact, not embedded — skip unless asked by name.
	if [ "$k" = megakernel ] && [ $# -eq 0 ]; then continue; fi
	out="testdata/${k%.cu}.ptx"
	LD_LIBRARY_PATH="$NVRTC_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
		NVRTC_SO="$NVRTC_LIB/libnvrtc.so.12" \
		python3 nvrtc_compile.py "$src" "$out" "$ARCH" "$CUDA_INC"
done

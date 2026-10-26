#!/usr/bin/env bash
# build_ptx.sh — regenerate the PTX embedded by aikit's CUDA path from the .cu sources.
#
# WHY THIS EXISTS
# ---------------
# The cgo-free CUDA device layer embeds PTX (go:embed) and hands it to the driver's JIT
# at run time (gpu/cuda.go: Device.CompileLibrary → cuModuleLoadData). That keeps the
# RUNTIME toolkit-free — libcuda and nothing else. But the PTX itself is a build-time
# artifact, and an artifact you cannot regenerate is an artifact you cannot review,
# audit, or change. Every shipped .ptx must be reproducible from a .cu in this repo by
# running this script. Regenerate — never hand-edit.
#
# WHY NVRTC AND NOT NVCC
# ----------------------
# nvcc drives a HOST compiler and includes host C++ headers, so it inherits the host
# toolchain's constraints — on a modern gcc, nvcc's `cicc` cannot parse libstdc++ and
# dies. NVRTC compiles CUDA C++ -> PTX with NO host compiler and no host headers, so it
# sidesteps the host-toolchain coupling entirely. (Same reasoning, same script, as
# goinfer/cuda/build_ptx.sh — this is the shared build step for the shared substrate.)
#
# LAYOUT
#   <dir>/foo.cu  ->  <dir>/testdata/foo.ptx     for each package dir below.
#
# USAGE
#   ./build_ptx.sh              # rebuild every kernel
#   ./build_ptx.sh gemv_w8a8    # rebuild one, by basename
#
# Override discovery with NVRTC_LIB (dir containing libnvrtc.so.12 AND
# libnvrtc-builtins.so.*). CUDA_INC (dir containing cuda_fp16.h) is optional — the
# current kernels use no CUDA headers — and is passed through when found.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${ARCH:-compute_75}" # Turing (RTX 2070). PTX is forward-compatible via the driver JIT.

# Package dirs that own .cu sources.
DIRS=(. anncuda)

find_first() { for c in "$@"; do [ -e "$c" ] && { echo "$c"; return; }; done; true; }

if [ -z "${NVRTC_LIB:-}" ]; then
	lib=$(find_first \
		/usr/local/cuda/lib64/libnvrtc.so.12 \
		/usr/lib64/libnvrtc.so.12 \
		"$HOME"/cuda-toolkit/targets/x86_64-linux/lib/libnvrtc.so.12 \
		"$HOME"/.venv*/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 \
		../.venv*/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 \
		./.venv*/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 \
		/tmp/cuda_extract/cuda_nvrtc/targets/x86_64-linux/lib/libnvrtc.so.12)
	[ -n "${lib:-}" ] && NVRTC_LIB="$(dirname "$lib")"
fi
if [ -z "${CUDA_INC:-}" ]; then
	hdr=$(find_first \
		/usr/local/cuda/include/cuda_fp16.h \
		/usr/include/cuda_fp16.h \
		"$HOME"/cuda-toolkit/targets/x86_64-linux/include/cuda_fp16.h \
		"$HOME"/.venv*/lib/python*/site-packages/nvidia/cuda_runtime/include/cuda_fp16.h \
		../.venv*/lib/python*/site-packages/nvidia/cuda_runtime/include/cuda_fp16.h \
		./.venv*/lib/python*/site-packages/nvidia/cuda_runtime/include/cuda_fp16.h \
		/tmp/cuda_extract/cuda_cudart/targets/x86_64-linux/include/cuda_fp16.h)
	[ -n "${hdr:-}" ] && CUDA_INC="$(dirname "$hdr")"
fi
if [ -z "${NVRTC_LIB:-}" ]; then
	echo "build_ptx: could not locate NVRTC (needs libnvrtc.so.12 + libnvrtc-builtins.so.*)." >&2
	echo "Install with:  pip install nvidia-cuda-nvrtc-cu12" >&2
	echo "then re-run with NVRTC_LIB=... $0" >&2
	exit 1
fi
echo "build_ptx: NVRTC_LIB=$NVRTC_LIB"
echo "build_ptx: CUDA_INC=${CUDA_INC:-<none — kernels use no CUDA headers>}"
echo "build_ptx: ARCH=$ARCH"

want=("$@")
built=0
for d in "${DIRS[@]}"; do
	shopt -s nullglob
	for src in "$d"/*.cu; do
		k="$(basename "$src" .cu)"
		if [ ${#want[@]} -gt 0 ]; then
			match=0
			for w in "${want[@]}"; do [ "${w%.cu}" = "$k" ] && match=1; done
			[ $match -eq 1 ] || continue
		fi
		mkdir -p "$d/testdata"
		out="$d/testdata/$k.ptx"
		LD_LIBRARY_PATH="$NVRTC_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
			NVRTC_SO="$NVRTC_LIB/libnvrtc.so.12" \
			python3 nvrtc_compile.py "$src" "$out" "$ARCH" ${CUDA_INC:+"$CUDA_INC"}
		built=$((built + 1))
	done
	shopt -u nullglob
done
[ $built -gt 0 ] || { echo "build_ptx: no kernel sources matched ${want[*]:-(all)}" >&2; exit 1; }

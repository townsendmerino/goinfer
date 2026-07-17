"""nvrtc_compile.py — compile a .cu to PTX with NVRTC (no host compiler involved).

Invoked by build_ptx.sh, which handles toolchain discovery. See that script for why the
CUDA kernels are built with NVRTC rather than nvcc (short version: nvcc drives a host
compiler and this box's gcc-15 libstdc++ breaks nvcc's cicc; NVRTC needs no host
compiler at all).

    python3 nvrtc_compile.py <src.cu> <out.ptx> <arch> [include-dir]

The NVRTC shared object is taken from $NVRTC_SO, else the loader path.
"""

import ctypes
import os
import sys

lib = ctypes.CDLL(os.environ.get("NVRTC_SO") or "libnvrtc.so.12")
lib.nvrtcGetErrorString.restype = ctypes.c_char_p


def fail(rc, where):
    print(f"NVRTC {where} failed: {lib.nvrtcGetErrorString(rc).decode()}", file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    src_path, out_path, arch = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(src_path, "rb") as f:
        src = f.read()

    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), src, src_path.encode(), 0, None, None)
    if rc != 0:
        fail(rc, "create")

    # -default-device: NVRTC has no host pass, so free functions need an implicit
    # __device__ rather than defaulting to __host__.
    opts = [f"--gpu-architecture={arch}".encode(), b"-default-device"]
    if len(sys.argv) > 4:
        opts.append(("--include-path=" + sys.argv[4]).encode())
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)

    # Always surface the log: NVRTC reports warnings on success too.
    logsz = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(logsz))
    if logsz.value > 1:
        log = ctypes.create_string_buffer(logsz.value)
        lib.nvrtcGetProgramLog(prog, log)
        if log.value.strip():
            print("=== NVRTC log ===\n" + log.value.decode(), file=sys.stderr)
    if rc != 0:
        fail(rc, "compile")

    ptxsz = ctypes.c_size_t()
    lib.nvrtcGetPTXSize(prog, ctypes.byref(ptxsz))
    ptx = ctypes.create_string_buffer(ptxsz.value)
    lib.nvrtcGetPTX(prog, ptx)
    with open(out_path, "wb") as f:
        f.write(ptx.raw.rstrip(b"\x00"))
    print(f"OK: wrote {out_path} ({ptxsz.value} bytes)")


if __name__ == "__main__":
    main()

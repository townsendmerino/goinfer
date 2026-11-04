// Package cuda is an opt-in, cgo-free native-CUDA backend for goinfer's resident decode path.
//
// The design splits in two:
//   - Layer A (driver plumbing): a 1:1 shim over github.com/eitamring/gocudrv that dlopens libcuda.so.1 at runtime, so
//     CGO_ENABLED=0 and the single-static-binary property hold. All context, alloc, memcpy, module-from-PTX, launch, stream and
//     event calls are here; gocudrv covers this except cooperative launch.
//   - Layer B (the compute): the kernel set in cuda/*.cu, each compiled to PTX by `go generate` (build_ptx.sh: NVRTC, not nvcc;
//     audited modules are pinned to NVRTC 12.6.85) and go:embed'd for driver-side JIT (cuModuleLoadDataEx). The census of
//     modules is the go:embed list in kernels.go.
//
// Fusion is per-op (fused_qkv.cu), not one all-in-one decode-layer kernel: the megakernel design in
// docs/completed/cuda-megakernel-spec.md did not ship (go/no-go: docs/completed/task-cuda-cgofree-spike.md).
//
// Build with `-tags cuda`. BuildResident builds a resident forward when the driver and the arch's features allow, and
// declines (ok=false) otherwise, in which case the decoder falls back to the staged/CPU path. Blank-importing this
// package is safe either way.
package cuda

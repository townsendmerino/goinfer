// Package cuda is an OPT-IN, cgo-free native-CUDA backend for goinfer's resident
// decode path — the Phase-1 skeleton of the megakernel spike.
//
// See docs/task-cuda-cgofree-spike.md (the go/no-go decision) and
// docs/cuda-megakernel-spec.md (the "implement exactly this" map).
//
// The design splits in two, and the split is the whole strategic point:
//
//   - Layer A (driver plumbing) — a 1:1 shim over github.com/eitamring/gocudrv:
//     dlopen libcuda.so.1 at runtime, so `CGO_ENABLED=0` and the single-static-
//     binary property hold. All context/alloc/memcpy/module-from-PTX/launch/
//     stream/event calls are here. gocudrv covers this EXCEPT cooperative launch.
//
//   - Layer B (the compute) — one fused decode-layer megakernel (megakernel.cu),
//     compiled to PTX by `go generate` (nvcc on the dev box) and go:embed'd, then
//     JIT'd on the target. This is the kernel WGSL structurally cannot express and
//     the only real work of the spike.
//
// Build with `-tags cuda`. BuildResident is live: it builds a resident forward when the driver
// and the arch's features allow, and declines (ok=false) otherwise, in which case the decoder
// falls back to the staged/CPU path. Blank-importing this package is safe either way.
//
// N-34: this described the Phase-2 state, in which BuildResident declined unconditionally
// "until the box wires the real kernel". It has not been true since v0.10.
package cuda

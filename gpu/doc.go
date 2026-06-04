// Package gpu is the OPTIONAL WebGPU (Metal / Vulkan / DX12) compute backend
// for goinfer's decoder and aikit's encoder matmuls, compiled only under the
// `gpu` build tag.
//
// It is the ONE place github.com/cogentcore/webgpu (cgo, bundling the
// wgpu-native Rust library) is allowed to appear. Every file except this doc
// carries `//go:build gpu`, and the backends register themselves through the
// decoder/encoder Backend registries on init — so the aikit and goinfer core
// modules never import webgpu, preserving their pure-Go / no-cgo promise.
//
// Usage: add a blank import of this package and build with `-tags gpu`:
//
//	import _ "github.com/townsendmerino/goinfer/gpu"
//
// then select the backend (decoder.Options{Backend: "webgpu"} /
// encoder NewBackend("webgpu")). Without the tag this module's
// implementation is absent and "webgpu" falls back to CPU with a note.
//
// Status: FOUNDATION cut. A single `dst = a·bᵀ` GEMM offloaded to the GPU
// (upload → dispatch → readback), with constant weights kept resident across
// calls. A single offloaded matmul is expected to LOSE to the CPU path on
// small/medium shapes because of host↔device transfer + kernel-launch
// overhead; the win only appears once the whole forward stays resident
// on-GPU across layers for large batches.
package gpu

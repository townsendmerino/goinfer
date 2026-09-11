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
// Status (corrected N-34 (09-02) — this said "FOUNDATION cut: a single dst = a·bᵀ GEMM
// offloaded" long after that stopped being true; restating a snapshot here just invites the
// same drift again, so this points at what stays current instead). This is a full resident
// decode runner, not a single offloaded matmul: MoE (routed + gated-shared expert), MLA,
// Mamba-2, Gated-DeltaNet (dense and MoE siblings), LoRA, and vision encoding are all
// implemented — see decoder/features.go's "webgpu" ResidentBackendFeatures entry for the exact,
// enforced capability set (the map a model is admitted against, so it cannot drift from what
// actually runs the way a status paragraph can) and the gpu/ test suite (-tags gpu) for
// end-to-end parity coverage per family.
package gpu

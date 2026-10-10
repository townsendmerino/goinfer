// Package gpu is the OPTIONAL WebGPU (Metal / Vulkan / DX12) compute backend
// for goinfer's decoder and aikit's encoder matmuls, compiled only under the
// `gpu` build tag.
//
// It is the ONE place github.com/oliverbestmann/webgpu (cgo, bundling the
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
// This is a full resident decode runner, not a single offloaded matmul.
// decoder/features.go's "webgpu" ResidentBackendFeatures entry is the enforced list of
// what it runs (the map a model is admitted against), and the gpu/ test suite
// (-tags gpu) holds the end-to-end parity coverage per family; do not restate a
// snapshot of either here. History: docs/code-notes/gpu.md#Package gpu: status paragraph.
package gpu

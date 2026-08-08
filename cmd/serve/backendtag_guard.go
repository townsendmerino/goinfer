//go:build cuda || gpu || metal

// This file exists only to FAIL THE BUILD when the pre-v0.10.0 command
//
//	go build -tags cuda|gpu|metal github.com/townsendmerino/goinfer/cmd/serve
//
// is run against the ROOT cmd/serve. Since v0.10.0 (audit M-19) the root binary is pure-Go
// and imports NO backend, so those build tags silently produced a CPU binary — the exact
// trap every pre-v0.10.0 doc/blog/note leads into. Build the submodule entrypoint instead:
//
//	go build -tags cuda  github.com/townsendmerino/goinfer/cuda/cmd/serve    # CUDA
//	go build -tags gpu   github.com/townsendmerino/goinfer/gpu/cmd/serve     # WebGPU
//	go build             github.com/townsendmerino/goinfer/metal/cmd/serve   # Metal (darwin)

package main

func init() {
	// Deliberate compile error — the identifier name is the guidance the compiler prints.
	_ = a_backend_build_tag_does_nothing_on_the_root_cmd_serve_since_v0_10_0__build_the_cuda_gpu_or_metal_cmd_serve_submodule_entrypoint_instead
}

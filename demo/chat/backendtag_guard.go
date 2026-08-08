//go:build cuda || gpu || metal

// Fails the build when the pre-v0.10.0 command `go build -tags cuda|gpu|metal .../demo/chat`
// is run against the ROOT demo/chat, which since v0.10.0 (audit M-19) is pure-Go and imports
// no backend — those tags silently produced a CPU binary. Build the submodule entrypoint:
//
//	go build -tags cuda  github.com/townsendmerino/goinfer/cuda/cmd/chat    # CUDA
//	go build -tags gpu   github.com/townsendmerino/goinfer/gpu/cmd/chat     # WebGPU
//	go build             github.com/townsendmerino/goinfer/metal/cmd/chat   # Metal (darwin)

package main

func init() {
	_ = a_backend_build_tag_does_nothing_on_the_root_demo_chat_since_v0_10_0__build_the_cuda_gpu_or_metal_cmd_chat_submodule_entrypoint_instead
}

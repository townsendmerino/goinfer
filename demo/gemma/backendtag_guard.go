//go:build cuda || gpu || metal

// Fails the build when a backend tag is passed to the ROOT demo/gemma, which since v0.10.0
// (audit M-19) is pure-Go and imports no backend — the tag silently produced a CPU binary.
// The gemma demo's accelerated build is Metal-only:
//
//	go build github.com/townsendmerino/goinfer/metal/cmd/gemma   # Metal (darwin)

package main

func init() {
	_ = a_backend_build_tag_does_nothing_on_the_root_demo_gemma_since_v0_10_0__build_the_metal_cmd_gemma_submodule_entrypoint_instead
}

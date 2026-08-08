//go:build gpu

// See backendtag_guard_cuda.go for the rationale (M-19: the root cmd/serve builds no backend,
// so `-tags gpu` on it silently yields a CPU binary). This makes it a build failure naming the
// WebGPU submodule entrypoint.
package main

const _ = "goinfer: `-tags gpu` does nothing on the root cmd/serve since v0.10.0 (it builds no backend). Build the submodule entrypoint instead: go build -tags gpu github.com/townsendmerino/goinfer/gpu/cmd/serve" + 1

//go:build metal

// See backendtag_guard_cuda.go for the rationale (M-19: the root cmd/serve builds no backend,
// so `-tags metal` on it silently yields a CPU binary). This makes it a build failure naming
// the Metal submodule entrypoint. Metal is GOOS=darwin-gated (no `-tags metal` required on the
// entrypoint itself), so the replacement command omits the tag.
package main

const _ = "goinfer: `-tags metal` does nothing on the root cmd/serve since v0.10.0 (it builds no backend). Build the submodule entrypoint instead (darwin): go build github.com/townsendmerino/goinfer/metal/cmd/serve" + 1

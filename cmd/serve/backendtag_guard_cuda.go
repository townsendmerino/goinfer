//go:build cuda

// Since v0.10.0 (audit M-19) the ROOT cmd/serve imports no backend — it is pure Go. A
// `-tags cuda` on it is accepted, exits 0, and silently builds a CPU binary: the pre-v0.10.0
// command that still lives in every cached README, blog post, and the v0.9.0 release page.
// This file turns that trap into a BUILD FAILURE whose text names the exact command to run
// instead. The `+ 1` is deliberate: it forces the compiler to echo the string literal —
// replacement command and all — verbatim into stderr (more readable than an undefined
// identifier, which cannot contain slashes). One file per tag so the message stays specific.
package main

const _ = "goinfer: `-tags cuda` does nothing on the root cmd/serve since v0.10.0 (it builds no backend). Build the submodule entrypoint instead: CGO_ENABLED=0 go build -tags cuda github.com/townsendmerino/goinfer/cuda/cmd/serve" + 1

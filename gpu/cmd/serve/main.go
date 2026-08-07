//go:build gpu

// Command serve is the WebGPU-accelerated build of goinfer's inference server.
//
// It is identical to the pure-Go root binary (github.com/townsendmerino/goinfer/cmd/serve)
// except that it blank-imports the opt-in WebGPU module first: that init() registers the
// "webgpu" decoder backend (decoder.RegisterBackend) and the resident SigLIP vision encoder
// (vision.RegisterResident). Living in the ./gpu submodule keeps cgo/webgpu OUT of the
// pure-Go root module graph (audit M-19). Run with `--backend webgpu`.
package main

import (
	_ "github.com/townsendmerino/goinfer/gpu"
	"github.com/townsendmerino/goinfer/internal/serveapp"
)

func main() { serveapp.Main() }

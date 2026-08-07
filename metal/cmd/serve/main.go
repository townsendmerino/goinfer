//go:build darwin

// Command serve is the Metal-accelerated (cgo-free) build of goinfer's inference server.
//
// Identical to the pure-Go root binary except it blank-imports the opt-in Metal module first,
// whose init() registers the "metal" decoder backend (decoder.RegisterBackend). Dense residency
// only; declines gracefully to the staged/CPU path when the device/kernels are unavailable or
// the weights aren't int8. Living in the ./metal submodule keeps Metal/purego OUT of the pure-Go
// root module graph (audit M-19). Run with `--backend metal`.
package main

import (
	"github.com/townsendmerino/goinfer/internal/serveapp"
	_ "github.com/townsendmerino/goinfer/metal"
)

func main() { serveapp.Main() }

//go:build darwin

// Command serve is the Metal-accelerated (cgo-free) build of goinfer's inference server.
//
// Identical to the pure-Go root binary except it blank-imports the opt-in Metal module first,
// whose init() registers the "metal" decoder backend (decoder.RegisterBackend). Both dense and
// MoE (including Gemma-4's parallel dense‖MoE) architectures can go resident; declines gracefully
// to the staged/CPU path when the device/kernels are unavailable or the arch/geometry isn't
// supported. Living in the ./metal submodule keeps Metal/purego OUT of the pure-Go
// root module graph. Run with `--backend metal`.
package main

import (
	_ "github.com/townsendmerino/goinfer/backend/metal"
	"github.com/townsendmerino/goinfer/internal/app/serveapp"
)

func main() { serveapp.Main() }

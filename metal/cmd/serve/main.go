//go:build darwin

// Command serve is the Metal-accelerated (cgo-free) build of goinfer's inference server.
//
// Identical to the pure-Go root binary except it blank-imports the opt-in Metal module first,
// whose init() registers the "metal" decoder backend (decoder.RegisterBackend). Both dense and
// MoE (including Gemma-4's parallel dense‖MoE) architectures can go resident; declines gracefully
// to the staged/CPU path when the device/kernels are unavailable or the arch/geometry isn't
// supported (N-11, audit-metal-2026-09-12.md: this used to say "dense residency only" and "the
// weights aren't int8" — Metal has no int8 GEMV and re-quantises int8int8 weights to W4A8 rather
// than declining them). Living in the ./metal submodule keeps Metal/purego OUT of the pure-Go
// root module graph (audit M-19). Run with `--backend metal`.
package main

import (
	"github.com/townsendmerino/goinfer/internal/serveapp"
	_ "github.com/townsendmerino/goinfer/metal"
)

func main() { serveapp.Main() }

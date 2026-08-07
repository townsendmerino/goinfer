//go:build cuda

// Command serve is the CUDA-accelerated (cgo-free) build of goinfer's inference server.
//
// Identical to the pure-Go root binary except it blank-imports the opt-in CUDA module first,
// whose init() registers the "cuda" decoder backend (decoder.RegisterBackend). Dense residency
// only; declines gracefully to the staged/CPU path when no NVIDIA driver is present. Living in
// the ./cuda submodule keeps gocudrv/libcuda OUT of the pure-Go root module graph (audit M-19).
// Run with `--backend cuda`.
package main

import (
	_ "github.com/townsendmerino/goinfer/cuda"
	"github.com/townsendmerino/goinfer/internal/serveapp"
)

func main() { serveapp.Main() }

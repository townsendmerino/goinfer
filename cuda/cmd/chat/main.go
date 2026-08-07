//go:build cuda

// Command chat is the CUDA-accelerated (cgo-free) build of goinfer's demo REPL (internal/chatapp).
// Blank-importing ./cuda registers the "cuda" decoder backend; living in the ./cuda submodule keeps
// gocudrv/libcuda out of the pure-Go root module graph (audit M-19). Run with `--backend cuda`.
package main

import (
	_ "github.com/townsendmerino/goinfer/cuda"
	"github.com/townsendmerino/goinfer/internal/chatapp"
)

func main() { chatapp.Main() }

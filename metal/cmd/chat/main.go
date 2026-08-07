//go:build darwin

// Command chat is the Metal-accelerated (cgo-free) build of goinfer's demo REPL (internal/chatapp).
// Blank-importing ./metal registers the "metal" decoder backend; living in the ./metal submodule
// keeps Metal/purego out of the pure-Go root module graph (audit M-19). Run with `--backend metal`.
package main

import (
	"github.com/townsendmerino/goinfer/internal/chatapp"
	_ "github.com/townsendmerino/goinfer/metal"
)

func main() { chatapp.Main() }

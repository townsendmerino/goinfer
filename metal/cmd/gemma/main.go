//go:build darwin

// Command gemma is the Metal-accelerated (cgo-free) build of the gemma demo CLI (internal/gemmaapp).
// Blank-importing ./metal registers the "metal" decoder backend; living in the ./metal submodule
// keeps Metal/purego out of the pure-Go root module graph (audit M-19). Run with `--backend metal`.
package main

import (
	"github.com/townsendmerino/goinfer/internal/gemmaapp"
	_ "github.com/townsendmerino/goinfer/metal"
)

func main() { gemmaapp.Main() }

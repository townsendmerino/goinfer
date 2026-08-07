// Command gemma is a pure-Go demo CLI that streams a local decoder-only LLM's
// completion to stdout (see internal/gemmaapp). CPU-only: no GPU/CUDA/Metal
// backend, keeping the pure-Go module graph clean (audit M-19). The Metal build
// lives in the ./metal submodule:
//
//	go build ./metal/cmd/gemma   # Metal (was: gemma -tags metal, darwin)
package main

import "github.com/townsendmerino/goinfer/internal/gemmaapp"

func main() { gemmaapp.Main() }

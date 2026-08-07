// Command chat is goinfer's interactive pure-Go demo REPL (see internal/chatapp).
//
// This root binary is CPU-only: it imports no GPU/CUDA/Metal backend, keeping the
// pure-Go module graph clean (audit M-19). The REPL logic — including the embedded
// single-file build (-tags embed / -tags prequant, see build-embed.sh) — lives in
// the importable internal/chatapp package. Accelerated builds are separate binaries
// that blank-import a backend first:
//
//	go build ./gpu/cmd/chat     # WebGPU  (was: chat -tags gpu)
//	go build ./cuda/cmd/chat    # CUDA    (was: chat -tags cuda)
//	go build ./metal/cmd/chat   # Metal   (was: chat -tags metal, darwin)
package main

import "github.com/townsendmerino/goinfer/internal/chatapp"

func main() { chatapp.Main() }

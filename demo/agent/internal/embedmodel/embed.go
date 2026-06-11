//go:build embed

// Package embedmodel is the one staging point for the baked-in GGUF, shared
// by both commands (//go:embed assets must live in the embedding package's
// own dir, so centralizing here avoids staging the model twice).
package embedmodel

import _ "embed"

// modelGGUF is stored UNCOMPRESSED — same rationale as demo/chat: q4 weights
// are high-entropy (zstd buys ~3%), and uncompressed lets the loader parse
// straight from the image-mapped slice with no heap copy. BUILD INPUT staged
// by build-embed.sh; gitignored, never committed.
//
//go:embed model.gguf
var modelGGUF []byte

// Bytes returns the baked-in GGUF and whether this build has one.
func Bytes() ([]byte, bool) { return modelGGUF, true }

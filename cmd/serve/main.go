// Command serve is goinfer's pure-Go OpenAI/Anthropic-compatible inference server.
//
// This root binary is CPU-only by design: it imports NONE of the opt-in GPU/CUDA/Metal
// backend modules, so the pure-Go module graph stays free of cgo/webgpu/purego/gocudrv
// (audit M-19 — a pure-Go consumer must not resolve the GPU dependency set). The server
// logic lives in the importable internal/serveapp package; the opt-in accelerated builds
// are separate binaries that blank-import a backend before calling the same entrypoint:
//
//	go build ./gpu/cmd/serve     # WebGPU  (was: serve -tags gpu)
//	go build ./cuda/cmd/serve    # CUDA    (was: serve -tags cuda)
//	go build ./metal/cmd/serve   # Metal   (was: serve -tags metal, darwin)
package main

import "github.com/townsendmerino/goinfer/internal/serveapp"

func main() { serveapp.Main() }

//go:build cuda

package main

// Blank-importing the opt-in cgo-free CUDA module runs its init(), which registers the
// "cuda" decoder backend (decoder.RegisterBackend). Built only under `-tags cuda` (with
// the go.work stitching the cuda submodule), so the default pure-Go build never pulls in
// gocudrv/libcuda. Dense residency only; declines gracefully to the staged/CPU path when
// no NVIDIA driver is present. `go run -tags cuda ./demo/chat --backend cuda …`.
import _ "github.com/townsendmerino/goinfer/cuda"

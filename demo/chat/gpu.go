//go:build gpu

package main

// Blank-importing the opt-in WebGPU module registers the "webgpu" decoder backend
// (decoder.RegisterBackend) via its init(), so `go run -tags gpu ./demo/chat
// --backend webgpu` decodes GPU-resident. Built only under `-tags gpu` (go.work
// stitches the gpu submodule); the default build stays pure-Go/cpu.
import _ "github.com/townsendmerino/goinfer/gpu"

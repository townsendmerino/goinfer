//go:build gpu

package main

// Blank-importing the opt-in WebGPU module runs its init(), which registers the
// "webgpu" decoder backend (decoder.RegisterBackend) and the resident SigLIP
// vision encoder (vision.RegisterResident). Built only under `-tags gpu` (with
// the go.work stitching the gpu submodule), so the default pure-Go serve binary
// never pulls in cgo/webgpu.
import _ "github.com/townsendmerino/goinfer/gpu"

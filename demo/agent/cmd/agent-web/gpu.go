//go:build gpu

package main

// Blank-importing the opt-in WebGPU module runs its init(), which registers the
// resident SigLIP vision encoder (vision.RegisterResident) so --vision-backend
// webgpu can run the tower on the device. Built only under `-tags gpu` (with the
// go.work stitching the gpu submodule); the default agent-web binary stays
// pure-Go (no cgo/webgpu).
import _ "github.com/townsendmerino/goinfer/gpu"

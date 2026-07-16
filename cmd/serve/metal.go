//go:build darwin && metal

package main

// Blank-importing the cgo-free Metal module runs its init(), registering the "metal" decoder
// backend (decoder.RegisterBackend). Built only under `-tags metal` on darwin (go.work stitches
// the metal submodule), so the default pure-Go serve binary never links Metal/purego. Dense
// residency only; declines gracefully to the staged/CPU path when the device/kernels are
// unavailable or the weights aren't int8. `serve -tags metal --backend metal …`.
import _ "github.com/townsendmerino/goinfer/metal"

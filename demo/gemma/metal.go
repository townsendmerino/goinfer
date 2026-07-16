//go:build darwin && metal

package main

// Blank-importing the cgo-free Metal module runs its init(), registering the "metal" decoder
// backend. Built only under `-tags metal` on darwin (go.work stitches the metal submodule),
// so the default pure-Go binary never links Metal/purego. Dense residency only; declines
// gracefully to the staged/CPU path when the device/kernels are unavailable or weights aren't
// int8. Run: `go run -tags metal ./demo/gemma --backend metal --quant int8int8 --model … …`.
import _ "github.com/townsendmerino/goinfer/metal"

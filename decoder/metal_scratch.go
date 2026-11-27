package decoder

import (
	"github.com/townsendmerino/aikit/linalg"

	"github.com/townsendmerino/goinfer/internal/runtime/memprice"
)

// MetalDecodeScratchBytes is what a Metal resident build of m leases on the scratch line for decode at slots expert slots per layer
// (0 = every expert resident), as ResidentNeedBytes takes them.
func (m *Model) MetalDecodeScratchBytes(slots int) int64 {
	if m == nil || m.w == nil || m.w.Arch == nil {
		return 0
	}
	return memprice.MetalDecodeScratchBytes(m.w, m.Program(), m.Quant(), slots, func(wm *linalg.WeightMat) string {
		return m.KQuantNative("metal", wm)
	})
}

// MetalDecodeRowWidths is decode's per-row activations as m's program declares them, each name at the widest layer that declares
// it: what a Metal build holds its decode rows to. A fresh map each call.
func (m *Model) MetalDecodeRowWidths() map[string]int {
	shape := m.Shape()
	return memprice.ComputeMetalDecodeRows(m.w.Arch, m.Program(), shape.Hidden, shape.Vocab).Widest
}

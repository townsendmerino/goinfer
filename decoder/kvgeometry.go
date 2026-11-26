package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// KVBlockRows is the number of positions in one logical KV block.
const KVBlockRows = kvmgr.BlockRows

const (
	KVWindowStepOK        = kvmgr.WindowStepOK
	KVWindowStepBelowBase = kvmgr.WindowStepBelowBase
	KVWindowStepOverCap   = kvmgr.WindowStepOverCap
)

// KVSlotRow is where logical block `block` of the sequence in KV slot `slot` lives while a backend stores KV per
// slot: the slot itself, at row block*KVBlockRows.
func KVSlotRow(slot, block int) (s, row int) { return kvmgr.SlotRow(slot, block) }

// KVWindowReusable reports whether a forward continuing at position prefix finds its sliding window held by a slot
// whose oldest resident position is base.
func KVWindowReusable(base, prefix, window int) bool {
	return kvmgr.WindowReusable(base, prefix, window)
}

// KVWindowNextBase computes the base before and after writing n rows starting at p0 into a windowed slot of capacity
// capN with window size window and rollback budget rollback.
func KVWindowNextBase(base, p0, n, window, capN, rollback int) (from, to, low int, status darch.KVWindowStepStatus) {
	return kvmgr.WindowNextBase(base, p0, n, window, capN, rollback)
}

// KVGeometry is the loaded model's KV shape (Architecture.KVGeometry); the zero value for a Model with no weights.
func (m *Model) KVGeometry() darch.KVGeometry {
	if m == nil || m.w == nil || m.w.Arch == nil {
		return darch.KVGeometry{}
	}
	return m.w.Arch.KVGeometry()
}

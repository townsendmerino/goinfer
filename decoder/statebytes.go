package decoder

import "github.com/townsendmerino/goinfer/internal/runtime/memprice"

// StateSpec is one buffer of a recurrent or conv mixer's state, for one KV slot, before any allocator's rounding.
type StateSpec = memprice.StateSpec

// StateSpecs is layer l's recurrent or conv state on this model, nil for a layer with none.
func (m *Model) StateSpecs(l int) []StateSpec { return memprice.StateSpecs(m.w.Arch, l) }

// CheckpointTotalBytes is CheckpointBuffers' sum: the checkpoint on a backend whose allocator rounds none of it.
func (q BuildQuery) CheckpointTotalBytes() int64 {
	var n int64
	for _, b := range q.CheckpointBuffers {
		n += b
	}
	return n
}

// StateTotalBytes is StateBuffers' sum: the state on a backend whose allocator rounds none of it.
func (q BuildQuery) StateTotalBytes() int64 {
	var n int64
	for _, b := range q.StateBuffers {
		n += b
	}
	return n
}

// stateBytes is one KV slot's state on backend, as its report prices it: Plan's state line.
func (m *Model) stateBytes(backend string) int64 {
	return m.buildDemand(backend).StateBytes
}

// checkpointBytes is the recurrent checkpoint on backend, as its report prices it: Plan.CheckpointBytes.
func (m *Model) checkpointBytes(backend string) (bytes int64, perSlot bool) {
	d := m.buildDemand(backend)
	return d.CheckpointBytes, d.CheckpointPerSlot
}

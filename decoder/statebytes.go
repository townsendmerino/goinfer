package decoder

import "github.com/townsendmerino/goinfer/internal/runtime/program"

// StateSpec is one buffer of a recurrent or conv mixer's state, for one KV slot, before any allocator's rounding.
type StateSpec struct {
	Name   string // "window" (the rolling conv window, full), "state" (the matrix state), or "window-q", -k and -v
	Floats int
}

// stateSpecs is layer l's state, its operations' per-slot buffers (layerBuffers): the one list a resident build allocates from and the
// plan prices from.
func stateSpecs(a *Architecture, l int) []StateSpec {
	var out []StateSpec
	for _, b := range layerBuffers(a, l) {
		if b.Unit == program.PerSlot {
			out = append(out, StateSpec{b.Name, b.Elems})
		}
	}
	return out
}

// StateSpecs is layer l's recurrent or conv state on this model, nil for a layer with none.
func (m *Model) StateSpecs(l int) []StateSpec { return stateSpecs(m.w.arch, l) }

// stateBuffers is one KV slot's recurrent and conv state for a, in bytes per buffer. It holds to what KVCache grows to
// (TestPlan_pricesRecurrentState).
func stateBuffers(a *Architecture) []int64 {
	var out []int64
	for l := range a.NumLayers {
		for _, s := range stateSpecs(a, l) {
			if s.Floats > 0 {
				out = append(out, int64(s.Floats)*4)
			}
		}
	}
	return out
}

// recurrentCheckpointBuffers is the state a resident build's rewind checkpoint copies, in bytes per buffer: what the operations mark
// checkpointed (Gated-DeltaNet's and Mamba-2's; KDA and short conv keep none).
func recurrentCheckpointBuffers(a *Architecture) []int64 {
	var out []int64
	for l := range a.NumLayers {
		for _, b := range layerBuffers(a, l) {
			if b.Unit == program.PerSlot && b.Checkpointed && b.Elems > 0 {
				out = append(out, int64(b.Elems)*4)
			}
		}
	}
	return out
}

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

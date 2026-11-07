package decoder

// StateSpec is one buffer of a recurrent or conv mixer's state, for one KV slot, before any allocator's rounding.
type StateSpec struct {
	Name   string // "window" (the rolling conv window, full), "state" (the matrix state), or "window-q", -k and -v
	Floats int
}

// stateSpecs is layer l's state: the one list a resident build allocates from and the plan prices from.
func stateSpecs(a *Architecture, l int) []StateSpec {
	switch {
	case a.qwen35 != nil && a.isLinearLayer(l):
		p := a.qwen35
		keyDim, valueDim := p.NumKeyHeads*p.KeyHeadDim, p.NumValueHeads*p.ValueHeadDim
		return []StateSpec{{"window", (p.ConvKernel - 1) * (2*keyDim + valueDim)}, {"state", p.NumValueHeads * p.KeyHeadDim * p.ValueHeadDim}}
	case a.lfm2 != nil && a.isConvLayer(l):
		return []StateSpec{{"window", (a.lfm2.ConvLCache - 1) * a.lfm2.ConvDim}}
	case a.kda != nil && a.isLinearLayer(l):
		p := a.kda
		win := (p.ConvKernel - 1) * p.NumHeads * p.HeadDim
		return []StateSpec{{"window-q", win}, {"window-k", win}, {"window-v", win}, {"state", p.NumHeads * p.HeadDim * p.HeadDim}}
	case a.granite != nil && a.isMambaLayer(l):
		g := a.granite
		return mamba2StateSpecs(mamba2Params{NHeads: g.NHeads, HeadDim: g.HeadDim, DState: g.DState, NGroups: g.NGroups, DConv: g.DConv})
	case a.nemotron != nil && a.nemotron.blockKind[l] == nemoMamba:
		n := a.nemotron
		return mamba2StateSpecs(mamba2Params{NHeads: n.NHeads, HeadDim: n.HeadDim, DState: n.DState, NGroups: n.NGroups, DConv: n.DConv})
	}
	return nil
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

// recurrentCheckpointBuffers is the state a resident build's rewind checkpoint copies, in bytes per buffer: Gated-DeltaNet's and Mamba-2's.
// A KDA or short-conv model keeps none.
func recurrentCheckpointBuffers(a *Architecture) []int64 {
	if a.kda != nil || a.lfm2 != nil {
		return nil
	}
	return stateBuffers(a)
}

// mamba2StateSpecs is a Mamba-2 layer's conv window and SSM state.
func mamba2StateSpecs(p mamba2Params) []StateSpec {
	return []StateSpec{{"window", (p.DConv - 1) * p.mixer().ConvDim()}, {"state", p.NHeads * p.HeadDim * p.DState}}
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

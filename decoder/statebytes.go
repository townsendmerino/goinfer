package decoder

// stateBuffers is one KV slot's recurrent and conv state for a, in bytes per buffer: each layer's rolling window, full, and its matrix
// state. It holds to what KVCache grows to (TestPlan_pricesRecurrentState) and to the mem.LineState buffers the resident builds allocate.
func stateBuffers(a *Architecture) []int64 {
	var out []int64
	add := func(floats ...int) {
		for _, n := range floats {
			if n > 0 {
				out = append(out, int64(n)*4)
			}
		}
	}
	for l := range a.NumLayers {
		switch {
		case a.qwen35 != nil && a.isLinearLayer(l):
			p := a.qwen35
			keyDim, valueDim := p.NumKeyHeads*p.KeyHeadDim, p.NumValueHeads*p.ValueHeadDim
			add((p.ConvKernel-1)*(2*keyDim+valueDim), p.NumValueHeads*p.KeyHeadDim*p.ValueHeadDim)
		case a.lfm2 != nil && a.isConvLayer(l):
			add((a.lfm2.ConvLCache - 1) * a.lfm2.ConvDim)
		case a.kda != nil && a.isLinearLayer(l):
			p := a.kda
			win := (p.ConvKernel - 1) * p.NumHeads * p.HeadDim // one per q, k and v stream
			add(win, win, win, p.NumHeads*p.HeadDim*p.HeadDim)
		case a.granite != nil && a.isMambaLayer(l):
			g := a.granite
			add(mamba2StateFloats(mamba2Params{NHeads: g.NHeads, HeadDim: g.HeadDim, DState: g.DState, NGroups: g.NGroups, DConv: g.DConv})...)
		case a.nemotron != nil && a.nemotron.blockKind[l] == nemoMamba:
			n := a.nemotron
			add(mamba2StateFloats(mamba2Params{NHeads: n.NHeads, HeadDim: n.HeadDim, DState: n.DState, NGroups: n.NGroups, DConv: n.DConv})...)
		}
	}
	return out
}

// mamba2StateFloats is a Mamba-2 layer's conv window and SSM state, in floats.
func mamba2StateFloats(p mamba2Params) []int {
	return []int{(p.DConv - 1) * p.mixer().ConvDim(), p.NHeads * p.HeadDim * p.DState}
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
	return buildDemandFor(m.Config(), Options{Backend: backend}, 0, 0, false).StateBytes
}

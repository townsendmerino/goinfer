package decoder

import "github.com/townsendmerino/goinfer/internal/kvmgr"

// The KV manager's types, under decoder names for a backend to use (the manager's own package is internal, so that
// its tests run apart from decoder's). docs/tasks/task-kv-architecture-2026-10.md §5.
type (
	// KVGeometry is a model's KV shape: its layer groups, the layers that read another layer's KV, and its
	// recurrent state.
	KVGeometry = kvmgr.Geometry
	// KVGroup is a set of layers with one KV row width and one retention, sharing a block table.
	KVGroup = kvmgr.Group
	// KVState is one kind of recurrent state a sequence carries.
	KVState = kvmgr.State
	// KVRetention is how long a group keeps a position.
	KVRetention = kvmgr.Retention
	// KVClass is the kernel class that wrote a KV row.
	KVClass = kvmgr.Class
	// KVClassMap is a backend's rule for which class a prefill pass writes.
	KVClassMap = kvmgr.ClassMap
)

// KVBlockRows is the number of positions in one logical KV block.
const KVBlockRows = kvmgr.BlockRows

const (
	KVRetainFull   = kvmgr.RetainFull
	KVRetainWindow = kvmgr.RetainWindow
	KVRetainChunk  = kvmgr.RetainChunk
)

// KVSlotRow is where logical block `block` of the sequence in KV slot `slot` lives while a backend stores KV per
// slot: the slot itself, at row block*KVBlockRows.
func KVSlotRow(slot, block int) (s, row int) { return kvmgr.SlotRow(slot, block) }

// KVGeometry derives the model's KV shape from its architecture. Layers are grouped by what decides a block table:
// the row width and the retention. It agrees with kvDimAt and kvPositionsAt, which price the same layers
// (TestKVGeometry_agreesWithThePricing).
func (a *Architecture) KVGeometry() KVGeometry {
	g := KVGeometry{NumLayers: a.NumLayers}
	states := map[string]int{}
	state := func(kind string, layer int) {
		i, ok := states[kind]
		if !ok {
			i = len(g.States)
			states[kind] = i
			g.States = append(g.States, KVState{Kind: kind})
		}
		g.States[i].Layers = append(g.States[i].Layers, layer)
	}
	for i := 0; i < a.NumLayers; i++ {
		switch {
		case a.isLinearLayer(i) && a.kda != nil:
			state("kda", i)
		case a.isLinearLayer(i):
			state("deltanet", i)
		case a.isMambaLayer(i), a.nemotron != nil && i < len(a.nemotron.blockKind) && a.nemotron.blockKind[i] == nemoMamba:
			state("mamba2", i)
		case a.isConvLayer(i):
			state("shortconv", i)
		}
		if a.hasNoAttentionKVAt(i) {
			continue
		}
		if src := a.gemma4KVSrcAt(i); src != i {
			if g.Shared == nil {
				g.Shared = map[int]int{}
			}
			g.Shared[i] = src
			continue
		}
		grp := KVGroup{RowWidth: a.kvDimAt(i), Latent: a.mla != nil,
			KEqualsV: a.gemma4 != nil && a.gemma4.KVShared && a.isGlobalLayer(i)}
		switch l4 := a.llama4; {
		case l4 != nil && l4.chunkSize > 0 && i < len(l4.useRope) && l4.useRope[i]:
			grp.Retention, grp.Span = KVRetainChunk, l4.chunkSize
		case a.SlidingWindow > 0 && !a.isGlobalLayer(i):
			grp.Retention, grp.Span = KVRetainWindow, a.SlidingWindow
		}
		at := -1
		for gi, have := range g.Groups {
			if have.Retention == grp.Retention && have.Span == grp.Span && have.RowWidth == grp.RowWidth && have.Latent == grp.Latent && have.KEqualsV == grp.KEqualsV {
				at = gi
				break
			}
		}
		if at < 0 {
			at = len(g.Groups)
			g.Groups = append(g.Groups, grp)
		}
		g.Groups[at].Layers = append(g.Groups[at].Layers, i)
	}
	return g
}

// KVGeometry is the loaded model's KV shape (Architecture.KVGeometry); the zero value for a Model with no weights.
func (m *Model) KVGeometry() KVGeometry {
	if m == nil || m.w == nil || m.w.arch == nil {
		return KVGeometry{}
	}
	return m.w.arch.KVGeometry()
}

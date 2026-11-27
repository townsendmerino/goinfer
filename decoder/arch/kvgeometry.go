package arch

import (
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/kvmgr"
)

// The KV manager's types, under decoder names for a backend to use (the manager's own package is internal, so that
// its tests run apart from decoder's).
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
	// KVSequenceState is the live KV state of one sequence, implemented by CPU sessions and GPU resident slots.
	KVSequenceState = kvmgr.SequenceState
	// KVWindowStepStatus classifies the outcome of advancing a windowed slot's base.
	KVWindowStepStatus = kvmgr.WindowStepStatus
)

const (
	KVRetainFull   = kvmgr.RetainFull
	KVRetainWindow = kvmgr.RetainWindow
	KVRetainChunk  = kvmgr.RetainChunk
)

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
		case a.IsLinearLayer(i) && a.KDA != nil:
			state("kda", i)
		case a.IsLinearLayer(i):
			state("deltanet", i)
		case a.IsMambaLayer(i), a.Nemotron != nil && i < len(a.Nemotron.BlockKind) && a.Nemotron.BlockKind[i] == archparts.NemoMamba:
			state("mamba2", i)
		case a.IsConvLayer(i):
			state("shortconv", i)
		case a.Inkling2 != nil:
			state("inkling2", i) // every visit also stores KV: its four conv windows are beside it
		}
		if a.HasNoAttentionKVAt(i) {
			continue
		}
		if src := a.Gemma4KVSrcAt(i); src != i {
			if g.Shared == nil {
				g.Shared = map[int]int{}
			}
			g.Shared[i] = src
			continue
		}
		grp := KVGroup{RowWidth: a.KVDimAt(i), Latent: a.MLA != nil,
			KEqualsV: a.Gemma4 != nil && a.Gemma4.KVShared && a.IsGlobalLayer(i)}
		switch l4 := a.Llama4; {
		case l4 != nil && l4.ChunkSize > 0 && i < len(l4.UseRope) && l4.UseRope[i]:
			grp.Retention, grp.Span = KVRetainChunk, l4.ChunkSize
		case a.SlidingWindow > 0 && !a.IsGlobalLayer(i) && a.Inkling2 == nil: // Inkling 2's local visits keep every position (KVPositionsAt)
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

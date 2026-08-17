package decoder

import "github.com/townsendmerino/aikit/linalg"

// The read-only view a backend needs to make a block drafter GPU-resident.
//
// WHY AN INTERFACE RATHER THAN EXPORTED FIELDS. The alternative was exporting `blockTrunk`'s
// fields so `cuda` could reach them, which publishes the drafter's LAYOUT as API — and
// `blockTrunk` is deliberately shared between DFlash and DSpark precisely so it can be
// refactored as more families land. This exports the CAPABILITY instead: a backend learns the
// geometry and gets the weights, and the struct behind them stays free to change.
//
// It also inverts the dependency the way `ResidentForward` already does — `decoder` declares
// what a backend may read, and `cuda` (or `metal`, or `gpu`) consumes it, rather than a backend
// importing a concrete drafter type and hard-coupling to one family.
//
// BOTH FAMILIES SATISFY IT FOR FREE. The methods are implemented once on `*blockTrunk`, which
// DFlashDrafter and DSparkDrafter both embed — so a resident path written against this interface
// serves either without a type switch. That is the property that made an interface worth writing
// now: there are two implementations to generalize over, not one to guess from.
//
// Everything returned is READ-ONLY. The WeightMat pointers alias the drafter's own storage
// (they are hundreds of MB; copying them to hand out would defeat the purpose), so a backend
// packs and uploads from them and must not write through them.
type BlockDrafterWeights interface {
	// DrafterGeometry describes the shapes a backend must allocate for.
	DrafterGeometry() DrafterGeometry
	// DrafterFC is the [hidden, nTaps*hidden] fusion projection over the concatenated tap
	// hidden states — the one projection with no counterpart in a normal decoder layer.
	DrafterFC() *linalg.WeightMat
	// DrafterHiddenNorm normalizes the fused context; DrafterFinalNorm ends the trunk.
	DrafterHiddenNorm() []float32
	DrafterFinalNorm() []float32
	// DrafterLayer is layer i's weights, 0 <= i < DrafterGeometry().Layers.
	DrafterLayer(i int) DrafterLayerWeights
	// BlockSize is the trained block width — how many positions the drafter drafts at once.
	// It lives on the concrete family rather than the shared trunk (the trunk runs whatever
	// width it is handed), and both families already expose it. How many of those positions
	// the TARGET then verifies is a separate, tunable choice — docs/spec/08 measures the
	// optimum as NARROWER than the trained width, which is why these are not the same number.
	BlockSize() int
}

// DrafterGeometry is the drafter's shape. It is deliberately flat rather than a reference to
// Architecture: a drafter is not a model (it has no embedding and no LM head — it borrows the
// target's), and handing a backend an Architecture would invite it to assume otherwise.
type DrafterGeometry struct {
	Layers       int
	Hidden       int
	NumHeads     int
	NumKVHeads   int
	HeadDim      int
	Intermediate int
	NormEps      float64
	// InvFreq is the RoPE inverse-frequency table, already built for this drafter's theta.
	InvFreq []float64
}

// DrafterLayerWeights is one trunk layer. The projections match a standard Qwen3-shaped layer,
// which is why a resident backend can reuse its existing per-layer kernels for everything except
// the attention MASK — the drafter's block is bidirectional (see attn_block_full).
type DrafterLayerWeights struct {
	Q, K, V, O     *linalg.WeightMat
	Gate, Up, Down *linalg.WeightMat
	// QNorm/KNorm are the per-head RMSNorm weights (Qwen3 qk-norm); nil when the family has none.
	QNorm, KNorm []float32
	// InputNorm normalizes the BLOCK only — never the context, whose K/V are projected from the
	// fused rows raw. Norming both is the natural-looking port and is wrong (see blockTrunk.layer).
	InputNorm    []float32
	PostAttnNorm []float32
}

// blockTrunk implements BlockDrafterWeights once, for every family that embeds it.

func (d *blockTrunk) DrafterGeometry() DrafterGeometry {
	return DrafterGeometry{
		Layers:       len(d.layers),
		Hidden:       d.hidden,
		NumHeads:     d.nHeads,
		NumKVHeads:   d.nKV,
		HeadDim:      d.headDim,
		Intermediate: d.inter,
		NormEps:      d.normEps,
		InvFreq:      d.invFreq,
	}
}

func (d *blockTrunk) DrafterFC() *linalg.WeightMat { return &d.fc }
func (d *blockTrunk) DrafterHiddenNorm() []float32 { return d.hiddenNorm }
func (d *blockTrunk) DrafterFinalNorm() []float32  { return d.finalNorm }

func (d *blockTrunk) DrafterLayer(i int) DrafterLayerWeights {
	l := &d.layers[i]
	return DrafterLayerWeights{
		Q: &l.q, K: &l.k, V: &l.v, O: &l.o,
		Gate: &l.gate, Up: &l.up, Down: &l.down,
		QNorm: l.qNorm, KNorm: l.kNorm,
		InputNorm: l.inputNorm, PostAttnNorm: l.postAttnNorm,
	}
}

// Compile-time proof that both families satisfy the interface through the shared trunk. If a
// third drafter lands and does NOT embed blockTrunk, this is where that shows up.
var (
	_ BlockDrafterWeights = (*DFlashDrafter)(nil)
	_ BlockDrafterWeights = (*DSparkDrafter)(nil)
)

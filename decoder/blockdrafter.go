package decoder

import "github.com/townsendmerino/aikit/linalg"

// BlockDrafterWeights is the read-only view a backend needs to make a block drafter GPU-resident.
//
// It is an interface rather than exported blockTrunk fields because exporting those would publish the drafter's layout
// as API, and blockTrunk is shared between DFlash and DSpark so that it can be refactored as more families land. A
// backend learns the geometry and gets the weights; the struct behind them stays free to change. It also inverts the
// dependency the way ResidentForward does: decoder declares what a backend may read and the backend consumes it.
//
// Both families satisfy it through blockTrunk, which DFlashDrafter and DSparkDrafter embed, so a resident path written
// against it serves either without a type switch.
//
// Everything returned is read-only. The WeightMat pointers alias the drafter's own storage (hundreds of MB), so a
// backend packs and uploads from them and must not write through them.
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
	// MaskTokenID is the trained token the drafter expects at unfilled block positions. It is on the interface because
	// passing any other id fails silently: the drafter still runs and stays lossless, and simply drafts badly.
	MaskTokenID() int
	// BlockSize is the trained block width: how many positions the drafter drafts at once. It lives on the concrete family
	// rather than the shared trunk, which runs whatever width it is handed. How many positions the target then verifies is
	// a separate, tunable choice, and the optimum is narrower than the trained width (docs/spec/08), so the two are not the
	// same number.
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

// DrafterResidentBytesEstimate approximates the VRAM a resident backend will claim uploading dw. It is computed rather
// than quoted so a caller (a fit guard pricing a --drafter attach before the target's own residency is built) gets a
// number that tracks the pairing actually loaded (docs/tasks/task-fit-to-hardware.md §2).
//
// It prices int8 specifically, not a generic multi-quant estimate as decoder/fitguard.go's quantBytesPerElem is for the
// main model: every drafter matrix is f32 on the host, and the only backend that hosts one (CUDA's AttachDrafter,
// cuda/drafter.go) packs every f32 matrix through packWeight's f32 branch, which is always int8 whatever the target's
// quant. Per-row scales (one f32 per row) are included; the norm vectors and RoPE table are f32 already and round to
// nothing beside the matrices.
//
// Not included: the verify and capture buffers NewBlockSpec allocates (a few MB at the default verify width, which the
// ctxCapMarginBytes and slotMarginBytes margins in cuda/resident.go absorb), and the drafter's own K/V, which scales
// with the target's context: see DrafterKVBytesPerPosition.
func DrafterResidentBytesEstimate(dw BlockDrafterWeights) int64 {
	int8Bytes := func(w *linalg.WeightMat) int64 {
		if w == nil {
			return 0
		}
		n, k := int64(w.Rows()), int64(w.Cols())
		return n*k + n*4 // packed int8 rows + one f32 scale per row
	}
	total := int8Bytes(dw.DrafterFC())
	geo := dw.DrafterGeometry()
	for i := 0; i < geo.Layers; i++ {
		l := dw.DrafterLayer(i)
		for _, w := range [...]*linalg.WeightMat{l.Q, l.K, l.V, l.O, l.Gate, l.Up, l.Down} {
			total += int8Bytes(w)
		}
	}
	return total
}

// DrafterKVBytesPerPosition returns the drafter's device K/V bytes per context position: layers x kvDim x 2 (K and V) x
// 4 (f32), matching cuda/drafter.go's d.kc/d.vc sizing. That K/V scales with the target's own resident context
// (ExtendContext sizes it at capRows = max(need+512, r.ctxCap)), so a multi-layer trunk at a large context is hundreds
// of MB, and DrafterResidentBytesEstimate cannot price it: it runs before the target's residency, and so its chosen
// ctx, exists.
//
// It is per position so each caller multiplies by the ctx it has in hand: resolveCtxCapFit by its candidate, the real
// build by the final ctxCap. Plan's chooseCtx only shrinks from the candidate it is asked with, so pricing the
// planning-time call against the candidate can only over-estimate (safe) or be exact.
func DrafterKVBytesPerPosition(dw BlockDrafterWeights) int64 {
	geo := dw.DrafterGeometry()
	kvDim := int64(geo.NumKVHeads) * int64(geo.HeadDim)
	return int64(geo.Layers) * kvDim * 2 /* K+V */ * 4 /* f32 */
}

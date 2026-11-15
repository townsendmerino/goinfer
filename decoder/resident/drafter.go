package resident

import "github.com/townsendmerino/aikit/linalg"

// BlockDrafterWeights is the read-only view a backend needs to make a block drafter GPU-resident.
//
// It is an interface rather than exported blockTrunk fields because exporting those would publish the drafter's layout
// as API, and blockTrunk is shared between DFlash and DSpark so that it can be refactored as more families land. A
// backend learns the geometry and gets the weights; the struct behind them stays free to change. It also inverts the
// dependency the way Forward does: decoder declares what a backend may read and the backend consumes it.
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

// BlockDrafter is a block drafter's trunk living on a backend's device, attached to that
// backend's resident target. Obtained from DrafterHost.AttachBlockDrafter.
type BlockDrafter interface {
	// FuseContext projects concatenated tap hidden states to the trunk's width and norms them.
	FuseContext(rows [][]float32) ([][]float32, error)
	// ExtendContext appends fused rows to the drafter's own K/V at the current context end. Incremental by contract: rebuilding the whole
	// context per block costs more, and the gap widens with length (docs/spec/08).
	ExtendContext(fused [][]float32) error
	// ContextLen is how many positions the drafter's context currently holds.
	ContextLen() int
	// TruncateContext drops context positions >= n — the rollback after a partial accept.
	TruncateContext(n int)
	// DraftBlock runs the trunk over one block, returning its output rows.
	DraftBlock(blockIn [][]float32) ([][]float32, error)
	// DraftTokens turns trunk rows into token ids using the TARGET's LM head — a block drafter
	// ships none of its own, which is why the pairing is fixed.
	DraftTokens(trunk [][]float32) ([]int, error)
}

// DrafterHost is a resident target that can host a block drafter and hand back the
// hidden states it needs. A backend implements this alongside Forward.
type DrafterHost interface {
	// AttachBlockDrafter uploads a drafter's weights to this target's device. It must reject a
	// geometry mismatch rather than resize: a drafter reads the target's residual stream, so a
	// mismatch is the wrong PAIRING, and the failure mode is a drafter that runs and drafts noise.
	AttachBlockDrafter(w BlockDrafterWeights) (BlockDrafter, error)
	// PrefillLastNArgmax verifies M rows in one batched pass, returning each row's argmax token
	// — all the accept decision reads. Only the ARGMAX must match a sequential Forward, not the
	// logits bit-for-bit, which is what lets the LM head be batched (docs/spec/08).
	PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error)
	// SetBatchedCapture arms the hidden-state seam for the whole batch: the next PrefillLastNArgmax records the residual at each named layer
	// for ALL its rows. The per-token seam costs a sync and a download per tap per token; this pays one per tap for the block.
	SetBatchedCapture(taps []int) error
	// BatchedCapture returns the last batched forward's rows as [tap][M*hidden].
	BatchedCapture() [][]float32
}

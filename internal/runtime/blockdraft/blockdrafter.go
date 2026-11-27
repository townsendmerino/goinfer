package blockdraft

import (
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder/resident"
)

// blockTrunk implements BlockDrafterWeights once, for every family that embeds it.

func (d *BlockTrunk) DrafterGeometry() resident.DrafterGeometry {
	return resident.DrafterGeometry{
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

// The weight accessors panic once the host weights are released (ErrDrafterReleased), and the matrix ones on a streamed
// drafter too (ErrDrafterStreamed), rather than hand out zero matrices.

func (d *BlockTrunk) DrafterFC() *linalg.WeightMat {
	d.mustHoldWeights()
	return &d.fc
}

func (d *BlockTrunk) DrafterHiddenNorm() []float32 {
	d.mustHoldNorms()
	return d.hiddenNorm
}

func (d *BlockTrunk) DrafterFinalNorm() []float32 {
	d.mustHoldNorms()
	return d.finalNorm
}

func (d *BlockTrunk) DrafterLayer(i int) resident.DrafterLayerWeights {
	d.mustHoldWeights()
	l := &d.layers[i]
	return resident.DrafterLayerWeights{
		Q: &l.q, K: &l.k, V: &l.v, O: &l.o,
		Gate: &l.gate, Up: &l.up, Down: &l.down,
		QNorm: l.qNorm, KNorm: l.kNorm,
		InputNorm: l.inputNorm, PostAttnNorm: l.postAttnNorm,
	}
}

// DrafterRows is a drafter matrix [Rows, Cols] that is not held: Read widens rows r0..r0+n-1 to f32 into dst (n*Cols long).
type DrafterRows struct {
	Rows, Cols int
	Read       func(r0, n int, dst []float32) error
}

// DrafterLayerRows is a layer's seven matrices as row readers, with its norms (DrafterLayerWeights' streamed form).
type DrafterLayerRows struct {
	Q, K, V, O, Gate, Up, Down DrafterRows
	QNorm, KNorm               []float32
	InputNorm, PostAttnNorm    []float32
}

// StreamedDrafterWeights is a drafter whose matrices may be row readers over its file rather than held (OpenDFlashDrafter).
// When Streamed reports true, the matrix accessors (DrafterFC, DrafterLayer) panic, and a backend's attach reads every matrix
// through DrafterFCRows and DrafterLayerRows instead.
type StreamedDrafterWeights interface {
	ReleasableDrafterWeights
	Streamed() bool
	DrafterFCRows() DrafterRows
	DrafterLayerRows(i int) DrafterLayerRows
}

// StreamingDrafterHost is a ResidentDrafterHost whose attach reads a streamed drafter's matrices through its row readers
// (StreamedDrafterWeights). NewBlockSpec refuses a streamed drafter on any other host, before anything is read.
type StreamingDrafterHost interface {
	resident.DrafterHost
	StreamsDrafterRows()
}

// ReleasableDrafterWeights is a BlockDrafterWeights whose host weights can be released once a resident backend holds its own
// copy (ReleaseHostWeights). A backend's attach checks HostWeightsReleased and refuses a released drafter: its matrices are
// zero, and an upload of them would draft noise that lossless verify only rejects. A family satisfies it only once its release
// also guards its own host drafting.
type ReleasableDrafterWeights interface {
	resident.BlockDrafterWeights
	HostWeightsReleased() bool
	ReleaseHostWeights() error
}

// Compile-time proof that both families satisfy the interfaces through the shared trunk. If a
// third drafter lands and does NOT embed blockTrunk, this is where that shows up.
var (
	_ StreamedDrafterWeights       = (*DFlash)(nil)
	_ resident.BlockDrafterWeights = (*DSpark)(nil) // not releasable: its own heads' host drafting has no release guard
)

// EstimateResidentBytes approximates the VRAM a resident backend will claim uploading dw. It is computed rather
// than quoted so a caller (a fit guard pricing a --drafter attach before the target's own residency is built) gets a
// number that tracks the pairing actually loaded.
//
// It prices int8 specifically, not a generic multi-quant estimate as decoder/fitguard.go's quantBytesPerElem is for the
// main model: every drafter matrix is f32 on the host, and the only backend that hosts one (CUDA's AttachDrafter,
// cuda/drafter.go) packs every f32 matrix through packWeight's f32 branch, which is always int8 whatever the target's
// quant. Per-row scales (one f32 per row) are included; the norm vectors and RoPE table are f32 already and round to
// nothing beside the matrices.
//
// Not included: the verify and capture buffers NewBlockSpec allocates (a few MB at the default verify width, which the
// CUDA context margin's decode allowance and slotMarginBytes absorb, cuda/margin.go and cuda/resident.go), and the drafter's own K/V, which scales
// with the target's context: see DrafterKVBytesPerPosition.
//
// Call it before a held drafter's release (ReleaseHostWeights): its matrix accessors then panic with ErrDrafterReleased. A
// streamed drafter's estimate reads only dimensions.
func EstimateResidentBytes(dw resident.BlockDrafterWeights) int64 {
	int8Bytes := func(w *linalg.WeightMat) int64 {
		if w == nil {
			return 0
		}
		n, k := int64(w.Rows()), int64(w.Cols())
		return n*k + n*4 // packed int8 rows + one f32 scale per row
	}
	geo := dw.DrafterGeometry()
	if s, ok := dw.(StreamedDrafterWeights); ok && s.Streamed() {
		rowBytes := func(r DrafterRows) int64 { return int64(r.Rows)*int64(r.Cols) + int64(r.Rows)*4 }
		total := rowBytes(s.DrafterFCRows())
		for i := 0; i < geo.Layers; i++ {
			l := s.DrafterLayerRows(i)
			for _, r := range [...]DrafterRows{l.Q, l.K, l.V, l.O, l.Gate, l.Up, l.Down} {
				total += rowBytes(r)
			}
		}
		return total
	}
	total := int8Bytes(dw.DrafterFC())
	for i := 0; i < geo.Layers; i++ {
		l := dw.DrafterLayer(i)
		for _, w := range [...]*linalg.WeightMat{l.Q, l.K, l.V, l.O, l.Gate, l.Up, l.Down} {
			total += int8Bytes(w)
		}
	}
	return total
}

// KVBytesPerPosition returns the drafter's device K/V bytes per context position: layers x kvDim x 2 (K and V) x
// 4 (f32), matching cuda/drafter.go's d.kc/d.vc sizing. That K/V scales with the target's own resident context
// (ExtendContext sizes it at capRows = max(need+512, r.ctxCap)), so a multi-layer trunk at a large context is hundreds
// of MB, and DrafterResidentBytesEstimate cannot price it: it runs before the target's residency, and so its chosen
// ctx, exists.
//
// It is per position so each caller multiplies by the ctx it has in hand: resolveCtxCapFit by its candidate, the real
// build by the final ctxCap. Plan's chooseCtx only shrinks from the candidate it is asked with, so pricing the
// planning-time call against the candidate can only over-estimate (safe) or be exact.
func KVBytesPerPosition(dw resident.BlockDrafterWeights) int64 {
	geo := dw.DrafterGeometry()
	kvDim := int64(geo.NumKVHeads) * int64(geo.HeadDim)
	return int64(geo.Layers) * kvDim * 2 /* K+V */ * 4 /* f32 */
}

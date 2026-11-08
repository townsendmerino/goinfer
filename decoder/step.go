package decoder

import (
	"context"
	"errors"
)

// stepIn is what a stepBatch's rows feed the model besides their token embeddings.
type stepIn uint8

const (
	inTokens         stepIn = iota // Emb at consecutive positions from Pos
	inRowsFn                       // N rows, which Rows hands over a chunk at a time
	inMRoPE                        // a rope position per row: RopePos for one row, MRoPE for several
	inMRoPEDeepstack               // inMRoPE plus Deep, injected over the image span Spans[0]
	inImage                        // one image span Spans[0], attended bidirectionally
	inImageBlocks                  // several image blocks, Spans
	inSpans                        // bidirectional spans, Spans
)

// stepOut is what a stepBatch returns. A one-row batch reads it for that row, a longer one for its last row, except the
// outEvery kinds, which return every row's.
type stepOut uint8

const (
	outLogits        stepOut = iota // Logits
	outArgmax                       // ID: the argmax id
	outTopK                         // TopK: the K best logits (K, Temperature, WantZ)
	outSample                       // ID: a device draw (Temperature, Seed, Draw)
	outNone                         // nothing: the rows' KV only
	outHidden                       // Hidden: the hidden state after the final norm
	outEveryLogits                  // Rows: every row's logits
	outEveryArgmax                  // IDs: every row's argmax id
	outEveryResidual                // Rows: every row's residual
)

// stepBatch is one forward call: rows of one sequence on the bound KV slot, from position Pos, with an input and an output
// kind; or, when Seqs is set, one decode row for each of several sequences on their own slots (MC3), and nothing else is read.
// Each field is read only by the kinds its comment names.
type stepBatch struct {
	Ctx  context.Context // the batched prefills' cancellation; nil is context.Background
	Pos  int
	Emb  [][]float32
	In   stepIn
	Out  stepOut
	N    int                          // inRowsFn: the row count
	Rows func(lo, hi int) [][]float32 // inRowsFn
	Mark int                          // with At, on a batched outLogits prefill: the position at which At runs (PrefillLastMarked)
	At   func(pos int)                // see Mark
	// RopePos, MRoPE, Deep, Spans: the rope and image inputs of the kinds named on stepIn.
	RopePos int
	MRoPE   [][3]int
	Deep    [][]float32
	Spans   [][2]int
	// Recurrent says the model holds recurrent state, which an m-RoPE prefill may build only where the resident reports
	// ResidentHybridMRoPEPrefill.
	Recurrent bool
	// K, Temperature, WantZ, Seed, Draw: outTopK's and outSample's arguments.
	K           int
	Temperature float64
	WantZ       bool
	Seed, Draw  uint64
	Seqs        []ResidentBatchSeq
}

// stepResult holds a step's output in the field its stepOut names; Batch holds a multi-sequence step's rows.
type stepResult struct {
	Logits []float32
	Hidden []float32
	Rows   [][]float32
	ID     int
	IDs    []int
	TopK   TopKRow
	Batch  []ResidentBatchOut
}

// errStepShape is stepOn's answer for a batch whose shape the resident does not offer (canStep false); the resident was not called.
var errStepShape = errors.New("decoder: the resident does not offer this step's shape")

// stepCall resolves b's shape on r to the one resident call that runs it, or nil when r does not offer that shape. It is the
// only place a forward shape is asserted on a resident: canStep and stepOn both read it.
func stepCall(r ResidentForward, b stepBatch) func() (stepResult, error) {
	ctx := b.Ctx
	if ctx == nil {
		ctx = context.Background()
	}
	var res stepResult
	if b.Seqs != nil {
		st, ok := r.(ResidentBatchStepper)
		if !ok {
			return nil
		}
		return func() (stepResult, error) {
			var err error
			res.Batch, err = st.StepBatch(b.Seqs)
			return res, err
		}
	}
	if b.In == inRowsFn {
		pr, ok := r.(ResidentPrefillRows)
		if !ok || b.Out != outLogits {
			return nil
		}
		return func() (stepResult, error) {
			var err error
			res.Logits, err = pr.PrefillLastRows(ctx, b.N, b.Pos, b.Rows)
			return res, err
		}
	}
	if len(b.Emb) == 0 {
		return nil
	}
	if len(b.Emb) == 1 && b.In == inTokens {
		e := b.Emb[0]
		switch b.Out {
		case outLogits:
			return func() (stepResult, error) {
				var err error
				res.Logits, err = r.Forward(e, b.Pos)
				return res, err
			}
		case outArgmax:
			g, ok := r.(ResidentGreedy)
			if !ok {
				return nil
			}
			return func() (stepResult, error) {
				var err error
				res.ID, err = g.ForwardArgmax(e, b.Pos)
				return res, err
			}
		case outTopK:
			tk, ok := r.(ResidentTopK)
			if !ok || !tk.TopKAvailable() {
				return nil
			}
			return func() (stepResult, error) {
				var err error
				res.TopK, err = tk.ForwardTopK(e, b.Pos, b.K, b.Temperature, b.WantZ)
				return res, err
			}
		case outSample:
			sm, ok := r.(ResidentSample)
			if !ok || !sm.SampleAvailable() {
				return nil
			}
			return func() (stepResult, error) {
				var err error
				res.ID, err = sm.ForwardSample(e, b.Pos, b.Temperature, b.Seed, b.Draw)
				return res, err
			}
		case outNone:
			kv, ok := r.(ResidentPrefillKV)
			if !ok {
				return nil
			}
			return func() (stepResult, error) { return res, kv.ForwardNoLogits(e, b.Pos) }
		}
	}
	if len(b.Emb) == 1 && b.In == inMRoPE && b.Out == outLogits {
		mr, ok := r.(ResidentMRoPE)
		if !ok {
			return nil
		}
		return func() (stepResult, error) {
			var err error
			res.Logits, err = mr.ForwardMRoPE(b.Emb[0], b.Pos, b.RopePos)
			return res, err
		}
	}
	switch b.In {
	case inTokens:
		switch b.Out {
		case outLogits:
			pf, ok := r.(Prefiller)
			if !ok {
				return nil
			}
			if b.At != nil {
				mp, ok := r.(ResidentPrefillMarked)
				if !ok {
					return nil
				}
				return func() (stepResult, error) {
					var err error
					res.Logits, err = mp.PrefillLastMarked(ctx, b.Emb, b.Pos, b.Mark, b.At)
					return res, err
				}
			}
			return func() (stepResult, error) {
				var err error
				res.Logits, err = pf.PrefillLast(ctx, b.Emb, b.Pos)
				return res, err
			}
		case outEveryLogits:
			return func() (stepResult, error) {
				var err error
				res.Rows, err = r.ForwardN(b.Emb, b.Pos)
				return res, err
			}
		case outArgmax:
			sa, ok := r.(ResidentSeedArgmax)
			if !ok {
				return nil
			}
			return func() (stepResult, error) {
				var err error
				res.ID, err = sa.PrefillSeedArgmax(b.Emb, b.Pos)
				return res, err
			}
		case outEveryArgmax:
			na, ok := r.(interface {
				PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error)
			})
			if !ok {
				return nil
			}
			return func() (stepResult, error) {
				var err error
				res.IDs, err = na.PrefillLastNArgmax(b.Emb, b.Pos)
				return res, err
			}
		case outHidden:
			hl, ok := r.(ResidentHiddenLast)
			if !ok {
				return nil
			}
			return func() (stepResult, error) {
				var err error
				res.Hidden, err = hl.HiddenLast(ctx, b.Emb, b.Pos)
				return res, err
			}
		case outEveryResidual:
			ra, ok := r.(ResidentResidualAll)
			if !ok {
				return nil
			}
			return func() (stepResult, error) {
				var err error
				res.Rows, err = ra.ResidualAll(ctx, b.Emb, b.Pos)
				return res, err
			}
		}
		return nil
	}
	if b.Out != outLogits {
		return nil
	}
	switch b.In {
	case inImage:
		ip, ok := r.(ResidentImagePrefill)
		if !ok || len(b.Spans) != 1 {
			return nil
		}
		return func() (stepResult, error) {
			var err error
			res.Logits, err = ip.PrefillImageLast(ctx, b.Emb, b.Pos, b.Spans[0][0], b.Spans[0][1])
			return res, err
		}
	case inImageBlocks:
		ib, ok := r.(ResidentImageBlocksPrefill)
		if !ok {
			return nil
		}
		return func() (stepResult, error) {
			var err error
			res.Logits, err = ib.PrefillImageBlocksLast(ctx, b.Emb, b.Pos, b.Spans)
			return res, err
		}
	case inSpans:
		sp, ok := r.(ResidentSpanPrefill)
		if !ok {
			return nil
		}
		return func() (stepResult, error) {
			var err error
			res.Logits, err = sp.PrefillSpansLast(ctx, b.Emb, b.Pos, b.Spans)
			return res, err
		}
	case inMRoPE, inMRoPEDeepstack:
		if b.Recurrent {
			hp, ok := r.(ResidentHybridMRoPEPrefill)
			if !ok || !hp.HybridMRoPEPrefill() {
				return nil
			}
		}
		if b.In == inMRoPE {
			mp, ok := r.(ResidentMRoPEPrefill)
			if !ok {
				return nil
			}
			return func() (stepResult, error) {
				var err error
				res.Logits, err = mp.PrefillMRoPELast(ctx, b.Emb, b.Pos, b.MRoPE)
				return res, err
			}
		}
		dp, ok := r.(ResidentMRoPEDeepstackPrefill)
		if !ok || len(b.Spans) != 1 {
			return nil
		}
		return func() (stepResult, error) {
			var err error
			lo, hi := b.Spans[0][0], b.Spans[0][1]
			res.Logits, err = dp.PrefillMRoPEDeepstackLast(ctx, b.Emb, b.Pos, b.MRoPE, b.Deep, lo, hi-lo)
			return res, err
		}
	}
	return nil
}

// canStep reports whether r offers b's shape: the type assertion, and the availability answer beside it where the shape has
// one (TopKAvailable, SampleAvailable, HybridMRoPEPrefill for a recurrent m-RoPE prefill). It calls no forward.
func canStep(r ResidentForward, b stepBatch) bool { return r != nil && stepCall(r, b) != nil }

// stepOn runs b on r as the one resident call today's callers make for its shape, or returns errStepShape, calling nothing,
// where canStep is false.
func stepOn(r ResidentForward, b stepBatch) (stepResult, error) {
	if r == nil {
		return stepResult{}, errStepShape
	}
	call := stepCall(r, b)
	if call == nil {
		return stepResult{}, errStepShape
	}
	return call()
}

// step is stepOn on m's resident. It does not claim or bind the resident: the caller holds it as for the call step makes.
func (m *Model) step(b stepBatch) (stepResult, error) { return stepOn(m.resident, b) }

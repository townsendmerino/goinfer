package decoder

import (
	"context"
	"errors"
)

// stepPass is the kind of resident call a stepBatch is, set by the caller and never inferred from the row count: a one-row
// prefill and a decode token are different calls on the resident.
type stepPass uint8

const (
	passNone   stepPass = iota // unset: no shape
	passToken                  // one decode token: Emb holds exactly one row
	passRows                   // one batched pass over Emb's rows (or inRowsFn's N), one row or more: prefill, verify, seed
	passSeqs                   // one decode row for each of several sequences on their own KV slots (MC3): Seqs
	passRagged                 // one or more rows for each of several sequences on their own KV slots: Ragged
)

// stepIn is what a stepBatch's rows feed the model besides their token embeddings.
type stepIn uint8

const (
	inTokens         stepIn = iota // Emb at consecutive positions from Pos
	inRowsFn                       // N rows, which Rows hands over a chunk at a time (passRows)
	inMRoPE                        // a rope position per row: RopePos for passToken, MRoPE for passRows
	inMRoPEDeepstack               // inMRoPE plus Deep, injected over the image span Spans[0] (passRows)
	inImage                        // one image span Spans[0], attended bidirectionally (passRows)
	inImageBlocks                  // several image blocks, Spans (passRows)
	inSpans                        // bidirectional spans, Spans (passRows)
)

// stepOut is what a stepBatch returns: for passToken that row's, for passRows its last row's, except the outEvery kinds,
// which return every row's.
type stepOut uint8

const (
	outLogits        stepOut = iota // Logits
	outArgmax                       // ID: the argmax id
	outTopK                         // TopK: the K best logits (K, Temperature, WantZ); passToken
	outSample                       // ID: a device draw (Temperature, Seed, Draw); passToken
	outNone                         // nothing: the row's KV only; passToken
	outHidden                       // Hidden: the hidden state after the final norm; passRows
	outEveryLogits                  // Rows: every row's logits; passRows
	outEveryArgmax                  // IDs: every row's argmax id; passRows
	outEveryResidual                // Rows: every row's residual; passRows
)

// stepBatch is one forward call: Pass says which kind; for passToken and passRows, rows of one sequence on the bound KV slot from
// position Pos with an input and an output kind; for passSeqs, Seqs alone. Each field is read only by the kinds its comment names.
type stepBatch struct {
	Pass stepPass
	Ctx  context.Context // passRows' cancellation; nil is context.Background
	Pos  int
	Emb  [][]float32
	In   stepIn
	Out  stepOut
	N    int                          // inRowsFn: the row count
	Rows func(lo, hi int) [][]float32 // inRowsFn
	Mark int                          // with At, on a passRows outLogits prefill: the position at which At runs (PrefillLastMarked)
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
	Ragged      []ResidentRaggedSeq
}

// stepResult holds a step's output in the field its stepOut names; Batch holds a passSeqs step's rows, Ragged a passRagged step's.
type stepResult struct {
	Logits []float32
	Hidden []float32
	Rows   [][]float32
	ID     int
	IDs    []int
	TopK   TopKRow
	Batch  []ResidentBatchOut
	Ragged []ResidentRaggedOut
}

// errStepShape is stepOn's answer for a batch whose shape the resident does not offer, or that names no shape (canStep false);
// the resident was not called.
var errStepShape = errors.New("decoder: the resident does not offer this step's shape")

// stepDo resolves b's shape on r to the one resident call that runs it, and makes that call when run is set; ok is false, and
// nothing is called, when b names no shape or r does not offer it. It is the only place a forward shape is asserted on a resident:
// canStep and stepOn both go through it, and it builds no closure, so a decode token's step allocates nothing of its own.
func stepDo(r any, sub ResidentSubsystems, b stepBatch, run bool) (stepResult, bool, error) {
	switch b.Pass {
	case passToken:
		if len(b.Emb) != 1 {
			return stepResult{}, false, nil
		}
		return tokenDo(r, sub.Decode, b, b.Emb[0], run)
	case passRows:
		ctx := b.Ctx
		if ctx == nil {
			ctx = context.Background()
		}
		return rowsDo(r, sub.Prefill, sub.Runtime, b, ctx, run)
	case passSeqs:
		if !sub.Decode.BatchStepperAvailable() || len(b.Seqs) == 0 {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Batch, err = sub.Decode.StepBatch(b.Seqs)
		return res, true, err
	case passRagged:
		if !sub.Decode.RaggedStepperAvailable() || len(b.Ragged) == 0 {
			return stepResult{}, false, nil
		}
		for _, q := range b.Ragged {
			if len(q.Embs) == 0 {
				return stepResult{}, false, nil
			}
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Ragged, err = sub.Decode.StepRagged(b.Ragged)
		return res, true, err
	}
	return stepResult{}, false, nil
}

// tokenDo is stepDo for passToken: the decode methods.
func tokenDo(r any, dec ResidentDecodeSubsystem, b stepBatch, e []float32, run bool) (stepResult, bool, error) {
	if b.In == inMRoPE {
		if !dec.MRoPEDecodeAvailable() || b.Out != outLogits {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Logits, err = dec.ForwardMRoPE(e, b.Pos, b.RopePos)
		return res, true, err
	}
	if b.In != inTokens {
		return stepResult{}, false, nil
	}
	switch b.Out {
	case outLogits:
		f, ok := r.(ResidentForward)
		if !ok {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Logits, err = f.Forward(e, b.Pos)
		return res, true, err
	case outArgmax:
		if !dec.GreedyAvailable() {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.ID, err = dec.ForwardArgmax(e, b.Pos)
		return res, true, err
	case outTopK:
		if !dec.TopKAvailable() {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.TopK, err = dec.ForwardTopK(e, b.Pos, b.K, b.Temperature, b.WantZ)
		return res, true, err
	case outSample:
		if !dec.SampleAvailable() {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.ID, err = dec.ForwardSample(e, b.Pos, b.Temperature, b.Seed, b.Draw)
		return res, true, err
	case outNone:
		if !dec.NoLogitsAvailable() {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		return stepResult{}, true, dec.ForwardNoLogits(e, b.Pos)
	}
	return stepResult{}, false, nil
}

// rowsDo is stepDo for passRows: the batched passes.
func rowsDo(r any, pf ResidentPrefillSubsystem, rt ResidentRuntimeSubsystem, b stepBatch, ctx context.Context, run bool) (stepResult, bool, error) {
	if b.In == inRowsFn {
		if !pf.PrefillRowsAvailable() || b.Out != outLogits || b.N <= 0 || b.Rows == nil || b.At != nil {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Logits, err = pf.PrefillLastRows(ctx, b.N, b.Pos, b.Rows)
		return res, true, err
	}
	if len(b.Emb) == 0 {
		return stepResult{}, false, nil
	}
	if b.In != inTokens {
		return mediaDo(pf, b, ctx, run)
	}
	switch b.Out {
	case outLogits:
		if b.At != nil {
			if !pf.PrefillMarkedAvailable() {
				return stepResult{}, false, nil
			}
			if !run {
				return stepResult{}, true, nil
			}
			var res stepResult
			var err error
			res.Logits, err = pf.PrefillLastMarked(ctx, b.Emb, b.Pos, b.Mark, b.At)
			return res, true, err
		}
		if !pf.PrefillAvailable() {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Logits, err = pf.PrefillLast(ctx, b.Emb, b.Pos)
		return res, true, err
	case outEveryLogits:
		f, ok := r.(ResidentForward)
		if !ok {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Rows, err = f.ForwardN(b.Emb, b.Pos)
		return res, true, err
	case outArgmax:
		if !rt.SeedArgmaxAvailable() {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.ID, err = rt.PrefillSeedArgmax(b.Emb, b.Pos)
		return res, true, err
	case outEveryArgmax:
		if !rt.LastNArgmaxAvailable() {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.IDs, err = rt.PrefillLastNArgmax(b.Emb, b.Pos)
		return res, true, err
	case outHidden:
		if !pf.HiddenLastAvailable() {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Hidden, err = pf.HiddenLast(ctx, b.Emb, b.Pos)
		return res, true, err
	case outEveryResidual:
		if !pf.ResidualAllAvailable() {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Rows, err = pf.ResidualAll(ctx, b.Emb, b.Pos)
		return res, true, err
	}
	return stepResult{}, false, nil
}

// mediaDo is rowsDo for the image and m-RoPE inputs, which return the last row's logits.
func mediaDo(pf ResidentPrefillSubsystem, b stepBatch, ctx context.Context, run bool) (stepResult, bool, error) {
	if b.Out != outLogits || b.At != nil {
		return stepResult{}, false, nil
	}
	switch b.In {
	case inImage:
		if !pf.ImagePrefillAvailable() || len(b.Spans) != 1 {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Logits, err = pf.PrefillImageLast(ctx, b.Emb, b.Pos, b.Spans[0][0], b.Spans[0][1])
		return res, true, err
	case inImageBlocks:
		if !pf.ImageBlocksPrefillAvailable() {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Logits, err = pf.PrefillImageBlocksLast(ctx, b.Emb, b.Pos, b.Spans)
		return res, true, err
	case inSpans:
		if !pf.SpanPrefillAvailable() {
			return stepResult{}, false, nil
		}
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Logits, err = pf.PrefillSpansLast(ctx, b.Emb, b.Pos, b.Spans)
		return res, true, err
	case inMRoPE, inMRoPEDeepstack:
		if b.Recurrent && !pf.HybridMRoPEPrefill() {
			return stepResult{}, false, nil
		}
		if b.In == inMRoPE {
			if !pf.MRoPEPrefillAvailable() {
				return stepResult{}, false, nil
			}
			if !run {
				return stepResult{}, true, nil
			}
			var res stepResult
			var err error
			res.Logits, err = pf.PrefillMRoPELast(ctx, b.Emb, b.Pos, b.MRoPE)
			return res, true, err
		}
		if !pf.MRoPEDeepstackPrefillAvailable() || len(b.Spans) != 1 {
			return stepResult{}, false, nil
		}
		lo, hi := b.Spans[0][0], b.Spans[0][1]
		if !run {
			return stepResult{}, true, nil
		}
		var res stepResult
		var err error
		res.Logits, err = pf.PrefillMRoPEDeepstackLast(ctx, b.Emb, b.Pos, b.MRoPE, b.Deep, lo, hi-lo)
		return res, true, err
	}
	return stepResult{}, false, nil
}

// canStep reports whether r (a resident, or a drafter host) offers b's shape: the capability, and the availability answer beside it where the shape has
// one (TopKAvailable, SampleAvailable, HybridMRoPEPrefill for a recurrent m-RoPE prefill). It calls no forward.
func canStep(r any, b stepBatch) bool {
	if r == nil {
		return false
	}
	_, ok, _ := stepDo(r, SubsystemsOf(r), b, false)
	return ok
}

// stepOn runs b on r as the one resident call today's callers make for its shape, or returns errStepShape, calling nothing,
// where canStep is false.
func stepOn(r any, b stepBatch) (stepResult, error) {
	if r == nil {
		return stepResult{}, errStepShape
	}
	res, ok, err := stepDo(r, SubsystemsOf(r), b, true)
	if !ok {
		return stepResult{}, errStepShape
	}
	return res, err
}

// step is stepOn on m's resident. It does not claim or bind the resident: the caller holds it as for the call step makes.
func (m *Model) step(b stepBatch) (stepResult, error) {
	if m.resident == nil {
		return stepResult{}, errStepShape
	}
	res, ok, err := stepDo(m.resident, m.subsystems(), b, true)
	if !ok {
		return stepResult{}, errStepShape
	}
	return res, err
}

// stepProbe is a one-row batch, stepProbeRows a rows function and stepProbeAt a mark callback, for asking canStep about a shape
// before there is a row: canStep reads none of them.
var (
	stepProbe     = [][]float32{nil}
	stepProbeRows = func(lo, hi int) [][]float32 { return nil }
	stepProbeAt   = func(pos int) {}
)

// stepProbeSpan is a one-span image input for asking canStep about the image and DeepStack shapes; canStep reads only its count.
var stepProbeSpan = [][2]int{{0, 1}}

// stepBatchRange is ResidentBatchStepper's BatchStepRange on r, the batch sizes its multi-sequence step serves; ok is false when r
// has no multi-sequence step (canStep refuses a passSeqs batch).
func stepBatchRange(r any) (lo, hi int, ok bool) {
	dec := SubsystemsOf(r).Decode
	if !dec.BatchStepperAvailable() {
		return 0, 0, false
	}
	lo, hi = dec.BatchStepRange()
	return lo, hi, true
}

// stepMarkAlign is ResidentPrefillMarked's PrefillMarkAlign on r, the pass width a marked prefill's mark must fall on; ok is false
// when r cannot mark a prefill (canStep refuses a passRows batch with At).
func stepMarkAlign(r any) (align int, ok bool) {
	pf := SubsystemsOf(r).Prefill
	if !pf.PrefillMarkedAvailable() {
		return 0, false
	}
	return pf.PrefillMarkAlign(), true
}

// canDecode reports whether m's resident offers the decode-token shape with output out (canStep on a passToken batch).
func (m *Model) canDecode(out stepOut) bool {
	if m.resident == nil {
		return false
	}
	_, ok, _ := stepDo(m.resident, m.subsystems(), stepBatch{Pass: passToken, Emb: stepProbe, Out: out}, false)
	return ok
}

package decoder

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

// stepLogResident implements every forward shape and logs each call as its method name and arguments, so a test can
// compare what step called with what today's callers call. Each call returns a distinct value the test checks came back.
type stepLogResident struct {
	calls   []string
	ctxs    []any // each ctx-taking call's ctx.Value(stepCtxKey{}), nil for a background context
	noTopK  bool  // TopKAvailable answers false
	noSamp  bool  // SampleAvailable answers false
	noHyMRP bool  // HybridMRoPEPrefill answers false
}

// stepCtxKey tags the caller's context, so a test sees which context reached the resident.
type stepCtxKey struct{}

func (f *stepLogResident) log(format string, a ...any) {
	f.calls = append(f.calls, fmt.Sprintf(format, a...))
}

func stepEmbs(es [][]float32) string {
	s := ""
	for _, e := range es {
		s += fmt.Sprint(e)
	}
	return s
}

func (f *stepLogResident) Forward(e []float32, pos int) ([]float32, error) {
	f.log("Forward %v %d", e, pos)
	return []float32{1}, nil
}
func (f *stepLogResident) ForwardN(es [][]float32, start int) ([][]float32, error) {
	f.log("ForwardN %s %d", stepEmbs(es), start)
	return [][]float32{{2}, {2}}, nil
}
func (f *stepLogResident) UploadKV(int, int, []float32, []float32) error { return nil }
func (f *stepLogResident) TruncateTo(int)                                {}
func (f *stepLogResident) Reset()                                        {}
func (f *stepLogResident) Close() error                                  { return nil }

func (f *stepLogResident) ForwardArgmax(e []float32, pos int) (int, error) {
	f.log("ForwardArgmax %v %d", e, pos)
	return 3, nil
}
func (f *stepLogResident) TopKAvailable() bool { return !f.noTopK }
func (f *stepLogResident) ForwardTopK(e []float32, pos, k int, t float64, z bool) (TopKRow, error) {
	f.log("ForwardTopK %v %d %d %v %v", e, pos, k, t, z)
	return TopKRow{IDs: []int32{4}}, nil
}
func (f *stepLogResident) SampleAvailable() bool { return !f.noSamp }
func (f *stepLogResident) ForwardSample(e []float32, pos int, t float64, seed, draw uint64) (int, error) {
	f.log("ForwardSample %v %d %v %d %d", e, pos, t, seed, draw)
	return 5, nil
}
func (f *stepLogResident) ForwardNoLogits(e []float32, pos int) error {
	f.log("ForwardNoLogits %v %d", e, pos)
	return nil
}
func (f *stepLogResident) ForwardMRoPE(e []float32, pos, rope int) ([]float32, error) {
	f.log("ForwardMRoPE %v %d %d", e, pos, rope)
	return []float32{6}, nil
}
func (f *stepLogResident) PrefillLast(ctx context.Context, es [][]float32, start int) ([]float32, error) {
	f.ctxs = append(f.ctxs, ctx.Value(stepCtxKey{}))
	f.log("PrefillLast %s %d", stepEmbs(es), start)
	return []float32{7}, nil
}
func (f *stepLogResident) PrefillMarkAlign() int { return 16 }
func (f *stepLogResident) PrefillLastMarked(ctx context.Context, es [][]float32, start, mark int, at func(int)) ([]float32, error) {
	f.ctxs = append(f.ctxs, ctx.Value(stepCtxKey{}))
	at(mark)
	f.log("PrefillLastMarked %s %d %d", stepEmbs(es), start, mark)
	return []float32{8}, nil
}
func (f *stepLogResident) PrefillLastRows(ctx context.Context, n, start int, rows func(lo, hi int) [][]float32) ([]float32, error) {
	f.ctxs = append(f.ctxs, ctx.Value(stepCtxKey{}))
	f.log("PrefillLastRows %d %d %s", n, start, stepEmbs(rows(0, n)))
	return []float32{9}, nil
}
func (f *stepLogResident) HiddenLast(ctx context.Context, es [][]float32, start int) ([]float32, error) {
	f.ctxs = append(f.ctxs, ctx.Value(stepCtxKey{}))
	f.log("HiddenLast %s %d", stepEmbs(es), start)
	return []float32{10}, nil
}
func (f *stepLogResident) ResidualAll(ctx context.Context, es [][]float32, start int) ([][]float32, error) {
	f.ctxs = append(f.ctxs, ctx.Value(stepCtxKey{}))
	f.log("ResidualAll %s %d", stepEmbs(es), start)
	return [][]float32{{11}}, nil
}
func (f *stepLogResident) PrefillImageLast(ctx context.Context, es [][]float32, start, lo, hi int) ([]float32, error) {
	f.ctxs = append(f.ctxs, ctx.Value(stepCtxKey{}))
	f.log("PrefillImageLast %s %d %d %d", stepEmbs(es), start, lo, hi)
	return []float32{12}, nil
}
func (f *stepLogResident) PrefillImageBlocksLast(ctx context.Context, es [][]float32, start int, blocks [][2]int) ([]float32, error) {
	f.ctxs = append(f.ctxs, ctx.Value(stepCtxKey{}))
	f.log("PrefillImageBlocksLast %s %d %v", stepEmbs(es), start, blocks)
	return []float32{13}, nil
}
func (f *stepLogResident) PrefillSpansLast(ctx context.Context, es [][]float32, start int, spans [][2]int) ([]float32, error) {
	f.ctxs = append(f.ctxs, ctx.Value(stepCtxKey{}))
	f.log("PrefillSpansLast %s %d %v", stepEmbs(es), start, spans)
	return []float32{14}, nil
}
func (f *stepLogResident) HybridMRoPEPrefill() bool { return !f.noHyMRP }
func (f *stepLogResident) PrefillMRoPELast(ctx context.Context, es [][]float32, start int, mp [][3]int) ([]float32, error) {
	f.ctxs = append(f.ctxs, ctx.Value(stepCtxKey{}))
	f.log("PrefillMRoPELast %s %d %v", stepEmbs(es), start, mp)
	return []float32{15}, nil
}
func (f *stepLogResident) PrefillMRoPEDeepstackLast(ctx context.Context, es [][]float32, start int, mp [][3]int, deep [][]float32, lo, n int) ([]float32, error) {
	f.ctxs = append(f.ctxs, ctx.Value(stepCtxKey{}))
	f.log("PrefillMRoPEDeepstackLast %s %d %v %s %d %d", stepEmbs(es), start, mp, stepEmbs(deep), lo, n)
	return []float32{16}, nil
}
func (f *stepLogResident) PrefillSeedArgmax(es [][]float32, start int) (int, error) {
	f.log("PrefillSeedArgmax %s %d", stepEmbs(es), start)
	return 17, nil
}
func (f *stepLogResident) PrefillLastNArgmax(es [][]float32, start int) ([]int, error) {
	f.log("PrefillLastNArgmax %s %d", stepEmbs(es), start)
	return []int{18, 18}, nil
}
func (f *stepLogResident) BatchStepRange() (int, int) { return 2, 8 }
func (f *stepLogResident) StepBatch(seqs []ResidentBatchSeq) ([]ResidentBatchOut, error) {
	s := ""
	for _, q := range seqs {
		s += fmt.Sprintf("{%d %d %v %v}", q.Slot, q.Pos, q.Emb, q.Draw != nil)
	}
	f.log("StepBatch %s", s)
	return []ResidentBatchOut{{ID: 19}}, nil
}

// TestStep_eachShapeMakesTodaysCall checks, for every forward shape, that step makes exactly the one resident call today's
// caller makes for it, with the same arguments, and hands back that call's result.
func TestStep_eachShapeMakesTodaysCall(t *testing.T) {
	ctx := context.WithValue(context.Background(), stepCtxKey{}, "caller")
	one := [][]float32{{0.5}}
	two := [][]float32{{0.5}, {0.25}}
	marked := -1
	draw := &ResidentBatchDraw{Temperature: 0.7, Seed: 1, Draw: 2}
	cases := []struct {
		name string
		b    stepBatch
		call string
		res  stepResult
	}{
		{"forward", stepBatch{Pass: passToken, Pos: 4, Emb: one}, "Forward [0.5] 4", stepResult{Logits: []float32{1}}},
		{"verify", stepBatch{Pass: passRows, Pos: 4, Emb: two, Out: outEveryLogits}, "ForwardN [0.5][0.25] 4", stepResult{Rows: [][]float32{{2}, {2}}}},
		{"greedy", stepBatch{Pass: passToken, Pos: 4, Emb: one, Out: outArgmax}, "ForwardArgmax [0.5] 4", stepResult{ID: 3}},
		{"top-k", stepBatch{Pass: passToken, Pos: 4, Emb: one, Out: outTopK, K: 40, Temperature: 0.7, WantZ: true}, "ForwardTopK [0.5] 4 40 0.7 true",
			stepResult{TopK: TopKRow{IDs: []int32{4}}}},
		{"sample", stepBatch{Pass: passToken, Pos: 4, Emb: one, Out: outSample, Temperature: 0.7, Seed: 9, Draw: 3}, "ForwardSample [0.5] 4 0.7 9 3", stepResult{ID: 5}},
		{"kv-only", stepBatch{Pass: passToken, Pos: 4, Emb: one, Out: outNone}, "ForwardNoLogits [0.5] 4", stepResult{}},
		{"m-rope decode", stepBatch{Pass: passToken, Pos: 4, Emb: one, In: inMRoPE, RopePos: 9}, "ForwardMRoPE [0.5] 4 9", stepResult{Logits: []float32{6}}},
		{"prefill", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: two}, "PrefillLast [0.5][0.25] 4", stepResult{Logits: []float32{7}}},
		{"prefill marked", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: two, Mark: 20, At: func(p int) { marked = p }},
			"PrefillLastMarked [0.5][0.25] 4 20", stepResult{Logits: []float32{8}}},
		{"prefill rows", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, In: inRowsFn, N: 2, Rows: func(lo, hi int) [][]float32 { return two[lo:hi] }},
			"PrefillLastRows 2 4 [0.5][0.25]", stepResult{Logits: []float32{9}}},
		{"hidden last", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: two, Out: outHidden}, "HiddenLast [0.5][0.25] 4", stepResult{Hidden: []float32{10}}},
		{"residual all", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: two, Out: outEveryResidual}, "ResidualAll [0.5][0.25] 4", stepResult{Rows: [][]float32{{11}}}},
		{"image", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: two, In: inImage, Spans: [][2]int{{1, 2}}}, "PrefillImageLast [0.5][0.25] 4 1 2",
			stepResult{Logits: []float32{12}}},
		{"image blocks", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: two, In: inImageBlocks, Spans: [][2]int{{0, 1}, {1, 2}}},
			"PrefillImageBlocksLast [0.5][0.25] 4 [[0 1] [1 2]]", stepResult{Logits: []float32{13}}},
		{"spans", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: two, In: inSpans, Spans: [][2]int{{0, 2}}}, "PrefillSpansLast [0.5][0.25] 4 [[0 2]]",
			stepResult{Logits: []float32{14}}},
		{"m-rope prefill", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: two, In: inMRoPE, MRoPE: [][3]int{{4, 4, 4}, {5, 5, 5}}},
			"PrefillMRoPELast [0.5][0.25] 4 [[4 4 4] [5 5 5]]", stepResult{Logits: []float32{15}}},
		{"m-rope deepstack", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: two, In: inMRoPEDeepstack, MRoPE: [][3]int{{4, 4, 4}, {5, 5, 5}},
			Deep: [][]float32{{0.125}}, Spans: [][2]int{{1, 2}}},
			"PrefillMRoPEDeepstackLast [0.5][0.25] 4 [[4 4 4] [5 5 5]] [0.125] 1 1", stepResult{Logits: []float32{16}}},
		{"seed argmax", stepBatch{Pass: passRows, Pos: 4, Emb: two, Out: outArgmax}, "PrefillSeedArgmax [0.5][0.25] 4", stepResult{ID: 17}},
		{"verify argmax", stepBatch{Pass: passRows, Pos: 4, Emb: two, Out: outEveryArgmax}, "PrefillLastNArgmax [0.5][0.25] 4", stepResult{IDs: []int{18, 18}}},
		// A one-row batched pass is still the batched call: passRows, never the decode method its row count would suggest.
		{"1-row prefill", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: one}, "PrefillLast [0.5] 4", stepResult{Logits: []float32{7}}},
		{"1-row prefill, nil ctx", stepBatch{Pass: passRows, Pos: 4, Emb: one}, "PrefillLast [0.5] 4", stepResult{Logits: []float32{7}}},
		{"1-row seed argmax", stepBatch{Pass: passRows, Pos: 4, Emb: one, Out: outArgmax}, "PrefillSeedArgmax [0.5] 4", stepResult{ID: 17}},
		{"1-row m-rope prefill", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: one, In: inMRoPE, MRoPE: [][3]int{{4, 4, 4}}},
			"PrefillMRoPELast [0.5] 4 [[4 4 4]]", stepResult{Logits: []float32{15}}},
		{"1-row marked prefill", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: one, Mark: 20, At: func(p int) { marked = p }},
			"PrefillLastMarked [0.5] 4 20", stepResult{Logits: []float32{8}}},
		{"recurrent m-rope prefill, hybrid", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: two, In: inMRoPE, Recurrent: true,
			MRoPE: [][3]int{{4, 4, 4}, {5, 5, 5}}}, "PrefillMRoPELast [0.5][0.25] 4 [[4 4 4] [5 5 5]]", stepResult{Logits: []float32{15}}},
		{"recurrent m-rope deepstack, hybrid", stepBatch{Pass: passRows, Ctx: ctx, Pos: 4, Emb: two, In: inMRoPEDeepstack, Recurrent: true,
			MRoPE: [][3]int{{4, 4, 4}, {5, 5, 5}}, Deep: [][]float32{{0.125}}, Spans: [][2]int{{1, 2}}},
			"PrefillMRoPEDeepstackLast [0.5][0.25] 4 [[4 4 4] [5 5 5]] [0.125] 1 1", stepResult{Logits: []float32{16}}},
		{"multi-sequence", stepBatch{Pass: passSeqs, Seqs: []ResidentBatchSeq{{Slot: 1, Pos: 3, Emb: []float32{0.5}}, {Slot: 2, Pos: 7, Emb: []float32{0.25}, Draw: draw}}},
			"StepBatch {1 3 [0.5] false}{2 7 [0.25] true}", stepResult{Batch: []ResidentBatchOut{{ID: 19}}}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := &stepLogResident{}
			if !canStep(f, c.b) {
				t.Fatalf("canStep refuses a shape the resident implements")
			}
			got, err := (&Model{resident: f}).step(c.b)
			if err != nil {
				t.Fatalf("step: %v", err)
			}
			if !reflect.DeepEqual(f.calls, []string{c.call}) {
				t.Fatalf("calls %q, want exactly %q", f.calls, c.call)
			}
			if !reflect.DeepEqual(got, c.res) {
				t.Fatalf("result %+v, want %+v", got, c.res)
			}
			// A context-taking call gets the caller's context, or a background one when the batch carries none.
			var want any
			if c.b.Ctx != nil {
				want = "caller"
			}
			for _, v := range f.ctxs {
				if v != want {
					t.Fatalf("the resident got context value %v, want %v", v, want)
				}
			}
			if c.b.Ctx != nil && len(f.ctxs) != 1 {
				t.Fatalf("a batch with a context made %d context-taking calls, want 1", len(f.ctxs))
			}
		})
	}
	if marked != 20 {
		t.Fatalf("the marked prefill's at was not the caller's (got %d)", marked)
	}
}

// stepPlainResident has only the required seam, so every optional shape is absent.
type stepPlainResident struct{ calls int }

func (p *stepPlainResident) Forward([]float32, int) ([]float32, error) {
	p.calls++
	return nil, nil
}
func (p *stepPlainResident) ForwardN([][]float32, int) ([][]float32, error) {
	p.calls++
	return nil, nil
}
func (p *stepPlainResident) UploadKV(int, int, []float32, []float32) error { return nil }
func (p *stepPlainResident) TruncateTo(int)                                {}
func (p *stepPlainResident) Reset()                                        {}
func (p *stepPlainResident) Close() error                                  { return nil }

// stepPrefillOnly is a Prefiller and nothing more optional.
type stepPrefillOnly struct{ stepPlainResident }

func (p *stepPrefillOnly) PrefillLast(context.Context, [][]float32, int) ([]float32, error) {
	p.calls++
	return nil, nil
}

// TestStep_absentShapeDeclines checks that canStep answers false, and stepOn returns errStepShape without calling the resident, for
// a shape the resident lacks or reports unavailable (TopKAvailable, SampleAvailable, HybridMRoPEPrefill), the answers today's
// callers read beside their type assertions.
func TestStep_absentShapeDeclines(t *testing.T) {
	one := [][]float32{{0.5}}
	two := [][]float32{{0.5}, {0.25}}
	rowsFn := func(lo, hi int) [][]float32 { return two[lo:hi] }
	optional := []stepBatch{
		{Pass: passToken, Emb: one, Out: outArgmax}, {Pass: passToken, Emb: one, Out: outTopK}, {Pass: passToken, Emb: one, Out: outSample},
		{Pass: passToken, Emb: one, Out: outNone}, {Pass: passToken, Emb: one, In: inMRoPE},
		{Pass: passRows, Emb: one}, {Pass: passRows, Emb: two}, {Pass: passRows, Emb: two, Mark: 1, At: func(int) {}},
		{Pass: passRows, In: inRowsFn, N: 2, Rows: rowsFn},
		{Pass: passRows, Emb: two, Out: outHidden}, {Pass: passRows, Emb: two, Out: outEveryResidual},
		{Pass: passRows, Emb: two, In: inImage, Spans: [][2]int{{0, 1}}}, {Pass: passRows, Emb: two, In: inImageBlocks},
		{Pass: passRows, Emb: two, In: inSpans}, {Pass: passRows, Emb: one, In: inMRoPE}, {Pass: passRows, Emb: two, In: inMRoPE},
		{Pass: passRows, Emb: two, In: inMRoPEDeepstack, Spans: [][2]int{{0, 1}}},
		{Pass: passRows, Emb: one, Out: outArgmax}, {Pass: passRows, Emb: two, Out: outArgmax}, {Pass: passRows, Emb: two, Out: outEveryArgmax},
		{Pass: passSeqs, Seqs: []ResidentBatchSeq{{}}},
	}
	p := &stepPlainResident{}
	for i, b := range optional {
		if canStep(p, b) {
			t.Errorf("case %d: canStep accepts a shape the resident lacks", i)
		}
		if _, err := stepOn(p, b); !errors.Is(err, errStepShape) {
			t.Errorf("case %d: stepOn err %v, want errStepShape", i, err)
		}
	}
	if p.calls != 0 {
		t.Fatalf("a declined step called the resident %d times", p.calls)
	}
	for _, c := range []struct {
		f *stepLogResident
		b stepBatch
	}{
		{&stepLogResident{noTopK: true}, stepBatch{Pass: passToken, Emb: one, Out: outTopK}},
		{&stepLogResident{noSamp: true}, stepBatch{Pass: passToken, Emb: one, Out: outSample}},
	} {
		if canStep(c.f, c.b) {
			t.Errorf("canStep accepts out kind %d the resident reports unavailable", c.b.Out)
		}
		if _, err := stepOn(c.f, c.b); !errors.Is(err, errStepShape) || len(c.f.calls) != 0 {
			t.Errorf("out kind %d reported unavailable: err %v, calls %q; want errStepShape and no call", c.b.Out, err, c.f.calls)
		}
	}
	deep := stepBatch{Pass: passRows, Emb: two, In: inMRoPEDeepstack, Spans: [][2]int{{0, 1}}}
	recDeep := deep
	recDeep.Recurrent = true
	for _, c := range []struct{ plain, recurrent stepBatch }{
		{stepBatch{Pass: passRows, Emb: two, In: inMRoPE}, stepBatch{Pass: passRows, Emb: two, In: inMRoPE, Recurrent: true}},
		{deep, recDeep},
	} {
		f := &stepLogResident{noHyMRP: true}
		if !canStep(f, c.plain) || canStep(f, c.recurrent) {
			t.Errorf("in kind %d: a recurrent m-RoPE prefill must need HybridMRoPEPrefill, a plain one must not", c.plain.In)
		}
		if _, err := stepOn(f, c.recurrent); !errors.Is(err, errStepShape) || len(f.calls) != 0 {
			t.Errorf("in kind %d: a recurrent prefill without HybridMRoPEPrefill: err %v, calls %q", c.plain.In, err, f.calls)
		}
	}
	// A batch that names no shape, or one its pass cannot run, declines on a resident that offers every shape.
	for i, b := range []stepBatch{
		{}, {Emb: one}, {Pass: passToken}, {Pass: passToken, Emb: two}, {Pass: passToken, Emb: one, Out: outHidden},
		{Pass: passRows}, {Pass: passRows, In: inRowsFn, Rows: rowsFn}, {Pass: passRows, In: inRowsFn, N: 2},
		{Pass: passRows, In: inRowsFn, N: 2, Rows: rowsFn, Out: outArgmax}, {Pass: passRows, Emb: two, Out: outTopK},
		{Pass: passRows, Emb: two, In: inImage}, {Pass: passRows, Emb: two, In: inMRoPEDeepstack},
		{Pass: passSeqs}, {Pass: passSeqs, Seqs: []ResidentBatchSeq{}},
	} {
		f := &stepLogResident{}
		if _, err := stepOn(f, b); !errors.Is(err, errStepShape) || len(f.calls) != 0 {
			t.Errorf("malformed batch %d: err %v, calls %q; want errStepShape and no call", i, err, f.calls)
		}
	}
	// A plain Prefiller without ResidentPrefillMarked declines a marked prefill instead of dropping the mark.
	pf := &stepPrefillOnly{}
	marked := stepBatch{Pass: passRows, Emb: two, Mark: 1, At: func(int) {}}
	if canStep(pf, marked) || !canStep(pf, stepBatch{Pass: passRows, Emb: two}) {
		t.Errorf("a Prefiller-only resident: a marked prefill must decline and a plain one must not")
	}
	if _, err := stepOn(pf, marked); !errors.Is(err, errStepShape) || pf.calls != 0 {
		t.Errorf("a marked prefill on a Prefiller-only resident: err %v, %d calls; want errStepShape and no call", err, pf.calls)
	}
}

// stepNoAllocResident answers each decode shape without allocating, so an allocation in a step is step's own.
type stepNoAllocResident struct{ stepPlainResident }

func (*stepNoAllocResident) ForwardArgmax([]float32, int) (int, error) { return 1, nil }
func (*stepNoAllocResident) TopKAvailable() bool                       { return true }
func (*stepNoAllocResident) ForwardTopK([]float32, int, int, float64, bool) (TopKRow, error) {
	return TopKRow{}, nil
}
func (*stepNoAllocResident) SampleAvailable() bool { return true }
func (*stepNoAllocResident) ForwardSample([]float32, int, float64, uint64, uint64) (int, error) {
	return 2, nil
}

// TestStep_decodeTokenAllocatesNothing: a decode token's step, with the caller's one-row slice reused across tokens as the decode
// loop does, allocates nothing, so moving the per-token calls behind step adds no garbage per token.
func TestStep_decodeTokenAllocatesNothing(t *testing.T) {
	m := &Model{resident: &stepNoAllocResident{}}
	emb := make([]float32, 8)
	var row [1][]float32
	for _, out := range []stepOut{outLogits, outArgmax, outTopK, outSample} {
		if a := testing.AllocsPerRun(100, func() {
			row[0] = emb
			if _, err := m.step(stepBatch{Pass: passToken, Emb: row[:], Pos: 3, Out: out}); err != nil {
				t.Fatal(err)
			}
		}); a != 0 {
			t.Errorf("out kind %d: %v allocations per decode step, want 0", out, a)
		}
	}
}

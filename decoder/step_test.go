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
	noTopK  bool // TopKAvailable answers false
	noSamp  bool // SampleAvailable answers false
	noHyMRP bool // HybridMRoPEPrefill answers false
}

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
func (f *stepLogResident) PrefillLast(_ context.Context, es [][]float32, start int) ([]float32, error) {
	f.log("PrefillLast %s %d", stepEmbs(es), start)
	return []float32{7}, nil
}
func (f *stepLogResident) PrefillMarkAlign() int { return 16 }
func (f *stepLogResident) PrefillLastMarked(_ context.Context, es [][]float32, start, mark int, at func(int)) ([]float32, error) {
	at(mark)
	f.log("PrefillLastMarked %s %d %d", stepEmbs(es), start, mark)
	return []float32{8}, nil
}
func (f *stepLogResident) PrefillLastRows(_ context.Context, n, start int, rows func(lo, hi int) [][]float32) ([]float32, error) {
	f.log("PrefillLastRows %d %d %s", n, start, stepEmbs(rows(0, n)))
	return []float32{9}, nil
}
func (f *stepLogResident) HiddenLast(_ context.Context, es [][]float32, start int) ([]float32, error) {
	f.log("HiddenLast %s %d", stepEmbs(es), start)
	return []float32{10}, nil
}
func (f *stepLogResident) ResidualAll(_ context.Context, es [][]float32, start int) ([][]float32, error) {
	f.log("ResidualAll %s %d", stepEmbs(es), start)
	return [][]float32{{11}}, nil
}
func (f *stepLogResident) PrefillImageLast(_ context.Context, es [][]float32, start, lo, hi int) ([]float32, error) {
	f.log("PrefillImageLast %s %d %d %d", stepEmbs(es), start, lo, hi)
	return []float32{12}, nil
}
func (f *stepLogResident) PrefillImageBlocksLast(_ context.Context, es [][]float32, start int, blocks [][2]int) ([]float32, error) {
	f.log("PrefillImageBlocksLast %s %d %v", stepEmbs(es), start, blocks)
	return []float32{13}, nil
}
func (f *stepLogResident) PrefillSpansLast(_ context.Context, es [][]float32, start int, spans [][2]int) ([]float32, error) {
	f.log("PrefillSpansLast %s %d %v", stepEmbs(es), start, spans)
	return []float32{14}, nil
}
func (f *stepLogResident) HybridMRoPEPrefill() bool { return !f.noHyMRP }
func (f *stepLogResident) PrefillMRoPELast(_ context.Context, es [][]float32, start int, mp [][3]int) ([]float32, error) {
	f.log("PrefillMRoPELast %s %d %v", stepEmbs(es), start, mp)
	return []float32{15}, nil
}
func (f *stepLogResident) PrefillMRoPEDeepstackLast(_ context.Context, es [][]float32, start int, mp [][3]int, deep [][]float32, lo, n int) ([]float32, error) {
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
	ctx := context.Background()
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
		{"forward", stepBatch{Pos: 4, Emb: one}, "Forward [0.5] 4", stepResult{Logits: []float32{1}}},
		{"verify", stepBatch{Pos: 4, Emb: two, Out: outEveryLogits}, "ForwardN [0.5][0.25] 4", stepResult{Rows: [][]float32{{2}, {2}}}},
		{"greedy", stepBatch{Pos: 4, Emb: one, Out: outArgmax}, "ForwardArgmax [0.5] 4", stepResult{ID: 3}},
		{"top-k", stepBatch{Pos: 4, Emb: one, Out: outTopK, K: 40, Temperature: 0.7, WantZ: true}, "ForwardTopK [0.5] 4 40 0.7 true",
			stepResult{TopK: TopKRow{IDs: []int32{4}}}},
		{"sample", stepBatch{Pos: 4, Emb: one, Out: outSample, Temperature: 0.7, Seed: 9, Draw: 3}, "ForwardSample [0.5] 4 0.7 9 3", stepResult{ID: 5}},
		{"kv-only", stepBatch{Pos: 4, Emb: one, Out: outNone}, "ForwardNoLogits [0.5] 4", stepResult{}},
		{"m-rope decode", stepBatch{Pos: 4, Emb: one, In: inMRoPE, RopePos: 9}, "ForwardMRoPE [0.5] 4 9", stepResult{Logits: []float32{6}}},
		{"prefill", stepBatch{Ctx: ctx, Pos: 4, Emb: two}, "PrefillLast [0.5][0.25] 4", stepResult{Logits: []float32{7}}},
		{"prefill marked", stepBatch{Ctx: ctx, Pos: 4, Emb: two, Mark: 20, At: func(p int) { marked = p }},
			"PrefillLastMarked [0.5][0.25] 4 20", stepResult{Logits: []float32{8}}},
		{"prefill rows", stepBatch{Ctx: ctx, Pos: 4, In: inRowsFn, N: 2, Rows: func(lo, hi int) [][]float32 { return two[lo:hi] }},
			"PrefillLastRows 2 4 [0.5][0.25]", stepResult{Logits: []float32{9}}},
		{"hidden last", stepBatch{Ctx: ctx, Pos: 4, Emb: two, Out: outHidden}, "HiddenLast [0.5][0.25] 4", stepResult{Hidden: []float32{10}}},
		{"residual all", stepBatch{Ctx: ctx, Pos: 4, Emb: two, Out: outEveryResidual}, "ResidualAll [0.5][0.25] 4", stepResult{Rows: [][]float32{{11}}}},
		{"image", stepBatch{Ctx: ctx, Pos: 4, Emb: two, In: inImage, Spans: [][2]int{{1, 2}}}, "PrefillImageLast [0.5][0.25] 4 1 2",
			stepResult{Logits: []float32{12}}},
		{"image blocks", stepBatch{Ctx: ctx, Pos: 4, Emb: two, In: inImageBlocks, Spans: [][2]int{{0, 1}, {1, 2}}},
			"PrefillImageBlocksLast [0.5][0.25] 4 [[0 1] [1 2]]", stepResult{Logits: []float32{13}}},
		{"spans", stepBatch{Ctx: ctx, Pos: 4, Emb: two, In: inSpans, Spans: [][2]int{{0, 2}}}, "PrefillSpansLast [0.5][0.25] 4 [[0 2]]",
			stepResult{Logits: []float32{14}}},
		{"m-rope prefill", stepBatch{Ctx: ctx, Pos: 4, Emb: two, In: inMRoPE, MRoPE: [][3]int{{4, 4, 4}, {5, 5, 5}}},
			"PrefillMRoPELast [0.5][0.25] 4 [[4 4 4] [5 5 5]]", stepResult{Logits: []float32{15}}},
		{"m-rope deepstack", stepBatch{Ctx: ctx, Pos: 4, Emb: two, In: inMRoPEDeepstack, MRoPE: [][3]int{{4, 4, 4}, {5, 5, 5}},
			Deep: [][]float32{{0.125}}, Spans: [][2]int{{1, 2}}},
			"PrefillMRoPEDeepstackLast [0.5][0.25] 4 [[4 4 4] [5 5 5]] [0.125] 1 1", stepResult{Logits: []float32{16}}},
		{"seed argmax", stepBatch{Pos: 4, Emb: two, Out: outArgmax}, "PrefillSeedArgmax [0.5][0.25] 4", stepResult{ID: 17}},
		{"verify argmax", stepBatch{Pos: 4, Emb: two, Out: outEveryArgmax}, "PrefillLastNArgmax [0.5][0.25] 4", stepResult{IDs: []int{18, 18}}},
		{"multi-sequence", stepBatch{Seqs: []ResidentBatchSeq{{Slot: 1, Pos: 3, Emb: []float32{0.5}}, {Slot: 2, Pos: 7, Emb: []float32{0.25}, Draw: draw}}},
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

// TestStep_absentShapeDeclines checks that canStep answers false, and stepOn returns errStepShape without calling the resident, for
// a shape the resident lacks or reports unavailable (TopKAvailable, SampleAvailable, HybridMRoPEPrefill), the answers today's
// callers read beside their type assertions.
func TestStep_absentShapeDeclines(t *testing.T) {
	one := [][]float32{{0.5}}
	two := [][]float32{{0.5}, {0.25}}
	optional := []stepBatch{
		{Emb: one, Out: outArgmax}, {Emb: one, Out: outTopK}, {Emb: one, Out: outSample}, {Emb: one, Out: outNone},
		{Emb: one, In: inMRoPE}, {Emb: two}, {Emb: two, Mark: 1, At: func(int) {}}, {In: inRowsFn, N: 2},
		{Emb: two, Out: outHidden}, {Emb: two, Out: outEveryResidual}, {Emb: two, In: inImage, Spans: [][2]int{{0, 1}}},
		{Emb: two, In: inImageBlocks}, {Emb: two, In: inSpans}, {Emb: two, In: inMRoPE}, {Emb: two, In: inMRoPEDeepstack, Spans: [][2]int{{0, 1}}},
		{Emb: two, Out: outArgmax}, {Emb: two, Out: outEveryArgmax}, {Seqs: []ResidentBatchSeq{{}}},
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
		{&stepLogResident{noTopK: true}, stepBatch{Emb: one, Out: outTopK}},
		{&stepLogResident{noSamp: true}, stepBatch{Emb: one, Out: outSample}},
	} {
		if canStep(c.f, c.b) {
			t.Errorf("canStep accepts out kind %d the resident reports unavailable", c.b.Out)
		}
	}
	if f := (&stepLogResident{noHyMRP: true}); canStep(f, stepBatch{Emb: two, In: inMRoPE, Recurrent: true}) || !canStep(f, stepBatch{Emb: two, In: inMRoPE}) {
		t.Errorf("a recurrent m-RoPE prefill must need HybridMRoPEPrefill, a plain one must not")
	}
	if _, err := stepOn(&stepLogResident{}, stepBatch{}); !errors.Is(err, errStepShape) {
		t.Errorf("an empty batch: err %v, want errStepShape", err)
	}
}

package decoder

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// callLogResident is fakeResident with the decode and verify shapes RF1-5 moves behind step (ResidentGreedy, ResidentTopK,
// ResidentSample, the argmax-only batched verify), logging every forward call by method and arguments. Each shape's answer is
// derived from the same row Forward returns, as the real backends' are.
type callLogResident struct {
	*fakeResident
	log []string
}

func (f *callLogResident) row(emb []float32, pos int) []float32 {
	r, _ := f.fakeResident.Forward(emb, pos)
	f.fakeResident.forwards--
	return r
}

func (f *callLogResident) Forward(emb []float32, pos int) ([]float32, error) {
	f.log = append(f.log, fmt.Sprintf("Forward %d", pos))
	return f.fakeResident.Forward(emb, pos)
}

func (f *callLogResident) ForwardN(embs [][]float32, start int) ([][]float32, error) {
	f.log = append(f.log, fmt.Sprintf("ForwardN %d+%d", start, len(embs)))
	out := make([][]float32, len(embs))
	for i, e := range embs {
		out[i] = f.row(e, start+i)
	}
	return out, nil
}

func (f *callLogResident) ForwardArgmax(emb []float32, pos int) (int, error) {
	f.log = append(f.log, fmt.Sprintf("ForwardArgmax %d", pos))
	return argmax(f.row(emb, pos)), nil
}

func (f *callLogResident) TopKAvailable() bool { return true }
func (f *callLogResident) ForwardTopK(emb []float32, pos, k int, temperature float64, wantZ bool) (TopKRow, error) {
	f.log = append(f.log, fmt.Sprintf("ForwardTopK %d k=%d t=%v z=%v", pos, k, temperature, wantZ))
	r := f.row(emb, pos)
	return TopKRow{Full: func() ([]float32, error) { return r, nil }}, nil
}

func (f *callLogResident) SampleAvailable() bool { return true }
func (f *callLogResident) ForwardSample(emb []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	f.log = append(f.log, fmt.Sprintf("ForwardSample %d t=%v seed=%d draw=%d", pos, temperature, seed, draw))
	return fakeDraw(f.row(emb, pos), seed, draw), nil
}

func (f *callLogResident) PrefillLastNArgmax(embs [][]float32, start int) ([]int, error) {
	f.log = append(f.log, fmt.Sprintf("PrefillLastNArgmax %d+%d", start, len(embs)))
	ids := make([]int, len(embs))
	for i, e := range embs {
		ids[i] = argmax(f.row(e, start+i))
	}
	return ids, nil
}

type callLogBackend struct {
	Backend
	rf   *callLogResident
	wrap func(*callLogResident) ResidentForward // the resident the model gets; nil is rf itself
}

func (b *callLogBackend) BuildResident(m *Model) (ResidentForward, bool, error) {
	_, _, _, _, _, _, vocab := m.Dims()
	b.rf = &callLogResident{fakeResident: &fakeResident{vocab: vocab}}
	if b.wrap != nil {
		return b.wrap(b.rf), true, nil
	}
	return b.rf, true, nil
}
func (b *callLogBackend) Close() error { return nil }

func loadWithCallLog(t *testing.T) (*Model, *callLogBackend) {
	t.Helper()
	return loadWithCallLogAs(t, nil)
}

// loadWithCallLogAs is loadWithCallLog with the resident wrap builds from the logging one.
func loadWithCallLogAs(t *testing.T, wrap func(*callLogResident) ResidentForward) (*Model, *callLogBackend) {
	t.Helper()
	be := &callLogBackend{wrap: wrap}
	RegisterBackend("fake-call-log", func() (Backend, error) {
		cpu, err := NewBackend("cpu")
		if err != nil {
			return nil, err
		}
		be.Backend = cpu
		return be, nil
	})
	m, err := Load(tinyFixture(t), Options{Backend: "fake-call-log"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { _ = m.Close() })
	if !m.ResidentActive() {
		t.Skip("fixture not resident-eligible")
	}
	return m, be
}

// TestStepCallers_decodeAndVerifyCallLog runs the decode loop's three device paths (greedy, the temperature-only device draw, the
// device top-K) and the n-gram speculative verify through a resident that logs its calls, and holds each log to the one the
// direct calls made before those callers moved behind step (recorded from them).
func TestStepCallers_decodeAndVerifyCallLog(t *testing.T) {
	prompt := []int{1, 2, 3, 1, 2, 3, 1, 2}
	cases := []struct {
		name string
		run  func(m *Model) ([]int, error)
		want string
	}{
		{"greedy", func(m *Model) ([]int, error) {
			return drainGen(m.Generate(context.Background(), prompt, 6, SamplingParams{}))
		},
			"Forward 0; Forward 1; Forward 2; Forward 3; Forward 4; Forward 5; Forward 6; Forward 7; ForwardArgmax 8; ForwardArgmax 9; ForwardArgmax 10; ForwardArgmax 11; ForwardArgmax 12; ForwardArgmax 13"},
		{"device sample", func(m *Model) ([]int, error) {
			return drainGen(m.Generate(context.Background(), prompt, 6, SamplingParams{Temperature: 0.7, Seed: 5}))
		},
			"Forward 0; Forward 1; Forward 2; Forward 3; Forward 4; Forward 5; Forward 6; Forward 7; ForwardSample 8 t=0.7 seed=5 draw=1; ForwardSample 9 t=0.7 seed=5 draw=2; ForwardSample 10 t=0.7 seed=5 draw=3; ForwardSample 11 t=0.7 seed=5 draw=4; ForwardSample 12 t=0.7 seed=5 draw=5; ForwardSample 13 t=0.7 seed=5 draw=6"},
		{"device top-k", func(m *Model) ([]int, error) {
			return drainGen(m.Generate(context.Background(), prompt, 6, SamplingParams{Temperature: 0.8, TopK: 5, TopP: 0.9, Seed: 7}))
		},
			"Forward 0; Forward 1; Forward 2; Forward 3; Forward 4; Forward 5; Forward 6; Forward 7; ForwardTopK 8 k=256 t=0.8 z=true; ForwardTopK 9 k=256 t=0.8 z=true; ForwardTopK 10 k=256 t=0.8 z=true; ForwardTopK 11 k=256 t=0.8 z=true; ForwardTopK 12 k=256 t=0.8 z=true; ForwardTopK 13 k=256 t=0.8 z=true"},
		{"ngram verify", func(m *Model) ([]int, error) {
			ch, g, err := m.GenerateNgramSpeculative(context.Background(), []int{5, 6, 7, 8, 9, 10, 11, 12, 5, 6}, 8, &NgramDrafter{MinMatch: 2, MaxMatch: 16}, 3, SamplingParams{})
			if err != nil {
				return nil, err
			}
			return drainGen(ch, g)
		},
			"Forward 0; Forward 1; Forward 2; Forward 3; Forward 4; Forward 5; Forward 6; Forward 7; Forward 8; Forward 9; ForwardArgmax 10; PrefillLastNArgmax 11+4; ForwardArgmax 14; ForwardArgmax 15; ForwardArgmax 16; ForwardArgmax 17"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m, be := loadWithCallLog(t)
			ids, err := c.run(m)
			if err != nil {
				t.Fatalf("generate: %v", err)
			}
			got := strings.Join(be.rf.log, "; ")
			t.Logf("ids %v\nlog %s", ids, got)
			if got != c.want {
				t.Errorf("call log\n got  %s\n want %s", got, c.want)
			}
		})
	}
}

func drainGen(ch <-chan int, g *Generation) ([]int, error) {
	var ids []int
	for id := range ch {
		ids = append(ids, id)
	}
	return ids, g.Err()
}

// callLogHost is a ResidentDrafterHost that answers every verify row with id and logs into the resident's call log, so a block
// speculation run's host calls and resident calls land in one order.
type callLogHost struct {
	rf *callLogResident
	id int
}

func (h *callLogHost) AttachBlockDrafter(BlockDrafterWeights) (ResidentBlockDrafter, error) {
	panic("not reached")
}
func (h *callLogHost) PrefillLastNArgmax(embs [][]float32, start int) ([]int, error) {
	h.rf.log = append(h.rf.log, fmt.Sprintf("host PrefillLastNArgmax %d+%d", start, len(embs)))
	ids := make([]int, len(embs))
	for i := range ids {
		ids[i] = h.id
	}
	return ids, nil
}
func (h *callLogHost) SetBatchedCapture([]int) error { return nil }
func (h *callLogHost) BatchedCapture() [][]float32   { return nil }

// TestStepCallers_blockSpecFallbackCallLog: a block drafter whose every draft the target rejects trips the acceptance guard, and
// the rest of the generation decodes on the resident's greedy step (BlockSpec's fallback), ending with the trailing token's
// forward; the call log is held to the one the direct calls made (recorded from them).
func TestStepCallers_blockSpecFallbackCallLog(t *testing.T) {
	m, be := loadWithCallLog(t)
	host := &callLogHost{rf: be.rf, id: 7}
	s := &BlockSpec{m: m, host: host, rd: blockspecEchoDrafter{id: 3}, dw: blockspecCommitStubWeights{blockspecStubWeights{blockSize: 8}}}
	out, rounds, err := s.generate([]int{1, 2, 3}, BlockSpecOptions{VerifyWidth: 4, MaxTokens: 14}, nil)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	got := strings.Join(be.rf.log, "; ")
	t.Logf("out %v rounds %d\nlog %s", out, rounds, got)
	const want = "host PrefillLastNArgmax 0+3; host PrefillLastNArgmax 3+4; host PrefillLastNArgmax 4+4; host PrefillLastNArgmax 5+4; host PrefillLastNArgmax 6+4; host PrefillLastNArgmax 7+4; host PrefillLastNArgmax 8+4; ForwardArgmax 9; ForwardArgmax 10; ForwardArgmax 11; ForwardArgmax 12; ForwardArgmax 13; ForwardArgmax 14; ForwardArgmax 15; ForwardArgmax 16"
	if got != want {
		t.Errorf("call log\n got  %s\n want %s", got, want)
	}
	if !strings.Contains(got, "ForwardArgmax") {
		t.Fatalf("the guard never tripped to plain decode: %s", got)
	}
}

// The prefill shapes RF1-6 moves behind step, each a wrapper adding one capability to the logging resident.
type (
	callLogPrefill struct{ *callLogResident }
	callLogRows    struct{ callLogPrefill }
	callLogKVOnly  struct{ *callLogResident }
	callLogBlocks  struct{ callLogPrefill }
)

func (f callLogPrefill) PrefillLast(_ context.Context, embs [][]float32, start int) ([]float32, error) {
	f.log = append(f.log, fmt.Sprintf("PrefillLast %d+%d", start, len(embs)))
	var last []float32
	for i, e := range embs {
		last = f.row(e, start+i)
	}
	return last, nil
}

func (f callLogRows) PrefillLastRows(_ context.Context, n, start int, rows func(lo, hi int) [][]float32) ([]float32, error) {
	f.log = append(f.log, fmt.Sprintf("PrefillLastRows %d+%d", start, n))
	var last []float32
	for i, e := range rows(0, n) {
		last = f.row(e, start+i)
	}
	return last, nil
}

func (f callLogKVOnly) ForwardNoLogits(emb []float32, pos int) error {
	f.log = append(f.log, fmt.Sprintf("ForwardNoLogits %d", pos))
	f.row(emb, pos)
	return nil
}

func (callLogBlocks) PrefillBlockRows() int              { return 4 }
func (callLogBlocks) PrefillScratchBytes(rows int) int64 { return int64(rows) }
func (callLogBlocks) PrefillTailExact() bool             { return true }

// TestStepCallers_prefillCallLog runs residentPrefillSeed's routes through a resident that logs its calls: one batched pass, the
// rows-at-a-time pass, the KV-only sequential loop of a short prompt, and the block split of a long one; each log is held to the
// one the direct calls made (recorded from them).
func TestStepCallers_prefillCallLog(t *testing.T) {
	long := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
	cases := []struct {
		name   string
		wrap   func(*callLogResident) ResidentForward
		prompt []int
		want   string
	}{
		{"batched", func(r *callLogResident) ResidentForward { return callLogPrefill{r} }, long,
			"PrefillLast 0+13; ForwardArgmax 13; ForwardArgmax 14"},
		{"rows", func(r *callLogResident) ResidentForward { return callLogRows{callLogPrefill{r}} }, long,
			"PrefillLastRows 0+13; ForwardArgmax 13; ForwardArgmax 14"},
		{"kv-only", func(r *callLogResident) ResidentForward { return callLogKVOnly{r} }, []int{1, 2, 3, 4, 5},
			"ForwardNoLogits 0; ForwardNoLogits 1; ForwardNoLogits 2; ForwardNoLogits 3; Forward 4; ForwardArgmax 5; ForwardArgmax 6"},
		{"blocks", func(r *callLogResident) ResidentForward { return callLogBlocks{callLogPrefill{r}} }, append(long, long...),
			"PrefillLast 0+4; PrefillLast 4+4; PrefillLast 8+4; PrefillLast 12+4; PrefillLast 16+10; ForwardArgmax 26; ForwardArgmax 27"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m, be := loadWithCallLogAs(t, c.wrap)
			if _, err := drainGen(m.Generate(context.Background(), c.prompt, 2, SamplingParams{})); err != nil {
				t.Fatalf("generate: %v", err)
			}
			got := strings.Join(be.rf.log, "; ")
			t.Logf("log %s", got)
			if got != c.want {
				t.Errorf("call log\n got  %s\n want %s", got, c.want)
			}
		})
	}
}

// callLogSeedHost is callLogHost that also seeds by ResidentSeedArgmax.
type callLogSeedHost struct{ *callLogHost }

func (h callLogSeedHost) PrefillSeedArgmax(embs [][]float32, start int) (int, error) {
	h.rf.log = append(h.rf.log, fmt.Sprintf("host PrefillSeedArgmax %d+%d", start, len(embs)))
	return h.id, nil
}

// TestStepCallers_blockSpecSeedCallLog: a drafter host that offers ResidentSeedArgmax seeds by it, not by the wide verify.
func TestStepCallers_blockSpecSeedCallLog(t *testing.T) {
	m, be := loadWithCallLog(t)
	host := callLogSeedHost{&callLogHost{rf: be.rf, id: 7}}
	s := &BlockSpec{m: m, host: host, rd: blockspecEchoDrafter{id: 7}, dw: blockspecCommitStubWeights{blockspecStubWeights{blockSize: 8}}}
	if _, _, err := s.generate([]int{1, 2, 3}, BlockSpecOptions{VerifyWidth: 4, MaxTokens: 1}, nil); err != nil {
		t.Fatalf("generate: %v", err)
	}
	got := strings.Join(be.rf.log, "; ")
	t.Logf("log %s", got)
	const want = "host PrefillSeedArgmax 0+3; host PrefillLastNArgmax 3+1"
	if got != want {
		t.Errorf("call log\n got  %s\n want %s", got, want)
	}
}

func (f *callLogResident) ForwardMRoPE(emb []float32, pos, rope int) ([]float32, error) {
	f.log = append(f.log, fmt.Sprintf("ForwardMRoPE %d rope=%d", pos, rope))
	return f.row(emb, pos), nil
}

// The multimodal and hidden-state shapes RF1-7 moves behind step, and a marked-capable resident.
type (
	callLogImage   struct{ *callLogResident }
	callLogMRoPEPf struct{ *callLogResident }
	callLogHidden  struct{ *callLogResident }
	callLogMarked  struct{ callLogPrefill }
	callLogSpans   struct{ *callLogResident }
)

func (f callLogSpans) PrefillSpansLast(_ context.Context, embs [][]float32, start int, spans [][2]int) ([]float32, error) {
	f.log = append(f.log, fmt.Sprintf("PrefillSpansLast %d+%d spans %v", start, len(embs), spans))
	var last []float32
	for i, e := range embs {
		last = f.row(e, start+i)
	}
	return last, nil
}

func (f callLogImage) PrefillImageLast(_ context.Context, embs [][]float32, start, lo, hi int) ([]float32, error) {
	f.log = append(f.log, fmt.Sprintf("PrefillImageLast %d+%d image [%d,%d)", start, len(embs), lo, hi))
	var last []float32
	for i, e := range embs {
		last = f.row(e, start+i)
	}
	return last, nil
}

func (f callLogMRoPEPf) PrefillMRoPELast(_ context.Context, embs [][]float32, start int, mp [][3]int) ([]float32, error) {
	f.log = append(f.log, fmt.Sprintf("PrefillMRoPELast %d+%d mrope %v..%v", start, len(embs), mp[0], mp[len(mp)-1]))
	var last []float32
	for i, e := range embs {
		last = f.row(e, start+i)
	}
	return last, nil
}

func (f callLogHidden) HiddenLast(_ context.Context, embs [][]float32, start int) ([]float32, error) {
	f.log = append(f.log, fmt.Sprintf("HiddenLast %d+%d", start, len(embs)))
	return append([]float32(nil), embs[len(embs)-1]...), nil
}

func (f callLogHidden) ResidualAll(_ context.Context, embs [][]float32, start int) ([][]float32, error) {
	f.log = append(f.log, fmt.Sprintf("ResidualAll %d+%d", start, len(embs)))
	out := make([][]float32, len(embs))
	for i, e := range embs {
		out[i] = append([]float32(nil), e...)
	}
	return out, nil
}

func (callLogMarked) PrefillMarkAlign() int { return 4 }
func (f callLogMarked) PrefillLastMarked(_ context.Context, embs [][]float32, start, mark int, at func(int)) ([]float32, error) {
	f.log = append(f.log, fmt.Sprintf("PrefillLastMarked %d+%d mark %d", start, len(embs), mark))
	at(mark)
	var last []float32
	for i, e := range embs {
		last = f.row(e, start+i)
	}
	return last, nil
}

// TestStepCallers_multimodalAndHiddenCallLog runs the image prefill (GenerateVL), the m-RoPE decode after the upload bridge and
// the m-RoPE prefill (GenerateQwenVL), HiddenLast, PromptHiddenAll, and a marked-capable resident's unmarked prefill, through a
// resident that logs its calls; each log is held to the one the direct calls made (recorded from them).
func TestStepCallers_multimodalAndHiddenCallLog(t *testing.T) {
	newRF := func(m *Model) *callLogResident {
		return &callLogResident{fakeResident: &fakeResident{vocab: m.w.arch.VocabSize}}
	}
	cases := []struct {
		name string
		run  func(t *testing.T) *callLogResident
		want string
	}{
		{"image prefill", func(t *testing.T) *callLogResident {
			m, g := loadGemma3VLTiny(t)
			rf := newRF(m)
			m.resident = callLogImage{rf}
			features := func() ([]float32, error) { return g.ImageFeatures, nil }
			if _, err := drainGen(m.GenerateVL(context.Background(), g.InputIDs, g.ImageTokenStart, g.MMTokens, 0, features, 3, SamplingParams{})); err != nil {
				t.Fatal(err)
			}
			return rf
		},
			"PrefillImageLast 0+9 image [3,7); Forward 9; Forward 10; Forward 11"},
		{"m-rope decode", func(t *testing.T) *callLogResident {
			m, g := loadQwen25VLTiny(t)
			rf := newRF(m)
			m.resident = rf
			features := func() ([]float32, error) { return g.ImageFeatures, nil }
			if _, err := drainGen(m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageStart, g.NImageTokens, 0, features, g.GridTHW, 2, g.ImageToken, 3, SamplingParams{})); err != nil {
				t.Fatal(err)
			}
			return rf
		},
			"ForwardMRoPE 13 rope=10; ForwardMRoPE 14 rope=11; ForwardMRoPE 15 rope=12"},
		{"m-rope prefill", func(t *testing.T) *callLogResident {
			m, g := loadQwen25VLTiny(t)
			rf := newRF(m)
			m.resident = callLogMRoPEPf{rf}
			features := func() ([]float32, error) { return g.ImageFeatures, nil }
			if _, err := drainGen(m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageStart, g.NImageTokens, 0, features, g.GridTHW, 2, g.ImageToken, 3, SamplingParams{})); err != nil {
				t.Fatal(err)
			}
			return rf
		},
			"PrefillMRoPELast 0+13 mrope [0 0 0]..[9 9 9]; ForwardMRoPE 13 rope=10; ForwardMRoPE 14 rope=11; ForwardMRoPE 15 rope=12"},
		{"m-rope reuse turn", func(t *testing.T) *callLogResident {
			m, g := loadQwen25VLTiny(t)
			rf := newRF(m)
			m.resident = rf
			features := func() ([]float32, error) { return g.ImageFeatures, nil }
			const imgHash = 7
			ids1, err := drainGen(m.GenerateQwenVL(context.Background(), g.InputIDs, g.ImageStart, g.NImageTokens, imgHash, features, g.GridTHW, 2, g.ImageToken, 3, SamplingParams{}))
			if err != nil {
				t.Fatal(err)
			}
			prompt2 := append(append(append([]int{}, g.InputIDs...), ids1...), g.InputIDs[len(g.InputIDs)-1])
			rf.log = nil // the second turn's calls: the reuse fast path's reseed and decode
			if _, err := drainGen(m.GenerateQwenVL(context.Background(), prompt2, g.ImageStart, g.NImageTokens, imgHash, features, g.GridTHW, 2, g.ImageToken, 3, SamplingParams{})); err != nil {
				t.Fatal(err)
			}
			return rf
		},
			"ForwardMRoPE 16 rope=13; ForwardMRoPE 17 rope=14; ForwardMRoPE 18 rope=15; ForwardMRoPE 19 rope=16"},
		{"hidden last", func(t *testing.T) *callLogResident {
			m, _ := loadWithCallLog(t)
			rf := newRF(m)
			m.resident = callLogHidden{rf}
			if _, err := m.HiddenLast([]int{1, 2, 3, 4}); err != nil {
				t.Fatal(err)
			}
			return rf
		},
			"HiddenLast 0+4"},
		{"residual all", func(t *testing.T) *callLogResident {
			m, _ := loadWithCallLog(t)
			rf := newRF(m)
			m.resident = callLogHidden{rf}
			if _, err := m.PromptHiddenAll(context.Background(), []int{1, 2, 3, 4}); err != nil {
				t.Fatal(err)
			}
			return rf
		},
			"ResidualAll 0+4"},
		{"marked-capable, no mark", func(t *testing.T) *callLogResident {
			m, be := loadWithCallLogAs(t, func(r *callLogResident) ResidentForward { return callLogMarked{callLogPrefill{r}} })
			if _, err := drainGen(m.Generate(context.Background(), []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, 2, SamplingParams{})); err != nil {
				t.Fatal(err)
			}
			return be.rf
		},
			"PrefillLast 0+13; ForwardArgmax 13; ForwardArgmax 14"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rf := c.run(t)
			got := strings.Join(rf.log, "; ")
			t.Logf("log %s", got)
			if got != c.want {
				t.Errorf("call log\n got  %s\n want %s", got, c.want)
			}
		})
	}
}

// TestStepCallers_gemma4SpanPrefillCallLog: a bidirectional Gemma 4 image turn on a resident with ResidentSpanPrefill takes one span
// prefill over the whole prompt, the image run its one block, then decodes on the resident from the prompt's end. Its fixture is
// generated (scripts/pin_gemma4_vl_bidir_tiny.py), so the call is checked by its form rather than a recorded log.
func TestStepCallers_gemma4SpanPrefillCallLog(t *testing.T) {
	m, g := loadGemma4VLBidirTiny(t)
	rf := &callLogResident{fakeResident: &fakeResident{vocab: m.w.arch.VocabSize}}
	m.resident = callLogSpans{rf}
	features := func() ([]float32, error) { return g.ImageFeatures, nil }
	ids, err := drainGen(m.GenerateGemma4VL(context.Background(), g.InputIDs, g.ImageTokenStart, g.NImageTokens, 0xC0FFEE, features, 3, SamplingParams{}))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{fmt.Sprintf("PrefillSpansLast 0+%d spans %v", len(g.InputIDs), [][2]int{{g.ImageTokenStart, g.ImageTokenStart + g.NImageTokens}})}
	for i := range len(rf.log) - 1 {
		want = append(want, fmt.Sprintf("Forward %d", len(g.InputIDs)+i))
	}
	if len(rf.log) < 2 || len(ids) == 0 || strings.Join(rf.log, "; ") != strings.Join(want, "; ") {
		t.Errorf("call log %q, want %q", rf.log, want)
	}
}

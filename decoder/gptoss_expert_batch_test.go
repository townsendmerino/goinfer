package decoder

import (
	"context"
	"math/rand/v2"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/testgate"
)

// TestGptOssExpertBatch_tinyLogitsBitIdentical decodes the committed tiny gpt-oss at int4 on the CPU with the expert gate/up
// batch off and on (w4a8BatchEnabled), through Generate's per-token walker, and requires every step's logits to match bit
// for bit, with the batch counted only when on.
func TestGptOssExpertBatch_tinyLogitsBitIdentical(t *testing.T) {
	m, err := Load("testdata/gptoss_tiny.gguf", Options{Backend: "cpu", Quant: "int4", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	capture := func(on bool) ([][]float32, int64) {
		testgate.SetForTest(t, &w4a8BatchEnabled, on)
		c0 := gptOssExpertBatchCalls.Load()
		var steps [][]float32
		sp := SamplingParams{Temperature: 0, LogitProcessor: func(_ []int, logits []float32) {
			steps = append(steps, slices.Clone(logits))
		}}
		out, g := m.Generate(context.Background(), []int{1, 5, 9, 17, 33, 65}, 12, sp)
		for range out {
		}
		if g.err != nil {
			t.Fatalf("generate: %v", g.err)
		}
		return steps, gptOssExpertBatchCalls.Load() - c0
	}
	off, offCalls := capture(false)
	on, onCalls := capture(true)
	if offCalls != 0 || onCalls == 0 {
		t.Fatalf("expert batch ran %d times off and %d on, want 0 and > 0", offCalls, onCalls)
	}
	if len(on) != len(off) || len(on) == 0 {
		t.Fatalf("captured %d steps on vs %d off", len(on), len(off))
	}
	for s := range off {
		for i := range off[s] {
			if on[s][i] != off[s][i] {
				t.Fatalf("step %d logit %d: batched %v != per-expert %v", s, i, on[s][i], off[s][i])
			}
		}
	}
}

// TestGptOssExpertBatch_fanOutBitIdentical runs gptOssMoEWith on a synthetic int4 layer large enough that the batched
// gate/up fans out across the experts' ops (8 ops of 256 rows at K=512), so shard boundaries fall inside and between ops,
// and requires the batched output to equal the per-expert loop's bit for bit.
func TestGptOssExpertBatch_fanOutBitIdentical(t *testing.T) {
	const hidden, inter, nExp, topK = 512, 256, 8, 4
	m, err := Load("testdata/gptoss_tiny.gguf", Options{Backend: "cpu", Quant: "int4", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	rng := rand.New(rand.NewPCG(20, 2880))
	fill := func(n int, scale float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * scale
		}
		return v
	}
	q4 := func(rows, cols int) linalg.WeightMat {
		return quantizeLayerWM(linalg.WrapF32(fill(rows*cols, 0.05), rows, cols), quantInt4, false, false)
	}
	lw := &LayerWeights{Router: linalg.WrapF32(fill(nExp*hidden, 0.1), nExp, hidden), RouterBias: fill(nExp, 0.1)}
	lw.Experts = make([]expertWeights, nExp)
	for e := range lw.Experts {
		lw.Experts[e] = expertWeights{
			Gate: q4(inter, hidden), Up: q4(inter, hidden), Down: q4(hidden, inter),
			GateBias: fill(inter, 0.1), UpBias: fill(inter, 0.1), DownBias: fill(hidden, 0.1),
		}
	}
	p := &ffnParams{Form: ffnGptOss, MoE: MoEConfig{NumExperts: nExp, TopK: topK, IntermediateDim: inter}, Alpha: 1.702, Limit: 7}
	h := fill(hidden, 1)

	t.Cleanup(func() { linalg.SetParallelWidth(0) })
	linalg.SetParallelWidth(8)
	run := func(on bool) ([]float32, int64) {
		testgate.SetForTest(t, &w4a8BatchEnabled, on)
		c0 := gptOssExpertBatchCalls.Load()
		out, err := m.gptOssMoEWith(h, lw, p, hidden, nil)
		if err != nil {
			t.Fatal(err)
		}
		return out, gptOssExpertBatchCalls.Load() - c0
	}
	want, offCalls := run(false)
	got, onCalls := run(true)
	if offCalls != 0 || onCalls != 1 {
		t.Fatalf("expert batch ran %d times off and %d on, want 0 and 1", offCalls, onCalls)
	}
	if 2*topK*inter*hidden < int4ParThreshold {
		t.Fatalf("the batch (%d MACs) would run serial; the test needs it to fan out", 2*topK*inter*hidden)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("out[%d]: batched %v != per-expert %v", i, got[i], want[i])
		}
	}
}

// TestGptOssMoEBatch_expertMajorEqualsPerRow runs gptOssMoEBatch over 24 rows of a synthetic int4 gpt-oss layer (8 experts, top 4,
// so each expert takes several rows and its matmuls run at M > 1) and requires every row to equal gptOssMoEWith on that row alone,
// bit for bit, with the per-row path's expert gate/up batch both off and on.
func TestGptOssMoEBatch_expertMajorEqualsPerRow(t *testing.T) {
	const hidden, inter, nExp, topK, n = 512, 256, 8, 4, 24
	m, err := Load("testdata/gptoss_tiny.gguf", Options{Backend: "cpu", Quant: "int4", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	rng := rand.New(rand.NewPCG(23, 11))
	fill := func(k int, scale float32) []float32 {
		v := make([]float32, k)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * scale
		}
		return v
	}
	q4 := func(rows, cols int) linalg.WeightMat {
		return quantizeLayerWM(linalg.WrapF32(fill(rows*cols, 0.05), rows, cols), quantInt4, false, false)
	}
	lw := &LayerWeights{Router: linalg.WrapF32(fill(nExp*hidden, 0.1), nExp, hidden), RouterBias: fill(nExp, 0.1)}
	lw.Experts = make([]expertWeights, nExp)
	for e := range lw.Experts {
		lw.Experts[e] = expertWeights{
			Gate: q4(inter, hidden), Up: q4(inter, hidden), Down: q4(hidden, inter),
			GateBias: fill(inter, 0.1), UpBias: fill(inter, 0.1), DownBias: fill(hidden, 0.1),
		}
	}
	p := &ffnParams{Form: ffnGptOss, MoE: MoEConfig{NumExperts: nExp, TopK: topK, IntermediateDim: inter}, Alpha: 1.702, Limit: 7}
	rows := fill(n*hidden, 1)

	got := make([]float32, n*hidden)
	r0 := gptOssExpertMajorRuns.Load()
	ok, err := m.gptOssMoEBatch(rows, n, lw, p, hidden, got, &moeBatchScratch{})
	if err != nil || !ok {
		t.Fatalf("gptOssMoEBatch: ok %v, err %v", ok, err)
	}
	if d := gptOssExpertMajorRuns.Load() - r0; d != 1 {
		t.Fatalf("expert-major ran %d times, want 1", d)
	}
	for _, batch := range []bool{false, true} {
		testgate.SetForTest(t, &w4a8BatchEnabled, batch)
		for r := range n {
			want, err := m.gptOssMoEWith(rows[r*hidden:(r+1)*hidden], lw, p, hidden, nil)
			if err != nil {
				t.Fatal(err)
			}
			for i, w := range want {
				if g := got[r*hidden+i]; g != w {
					t.Fatalf("per-row gate/up batch %v: row %d out[%d]: expert-major %v != per-row %v", batch, r, i, g, w)
				}
			}
		}
	}
}

// TestGptOssPrefill_batchedEqualsSequential prefills the tiny gpt-oss (window layers on the KV ring, learned sinks) past twice its
// window through prefillLogits and through the per-token walker, with exact attention, and requires the last row's logits and the
// next 8 greedy steps' logits to match bit for bit. It runs once with the expert-major FFN and once with it off (the batched
// walker's per-row FFN), and requires the batched walker and, in the first run, the expert-major path to have run.
func TestGptOssPrefill_batchedEqualsSequential(t *testing.T) {
	for _, em := range []string{"1", "0"} {
		t.Run("expert-major="+em, func(t *testing.T) {
			m, err := Load("testdata/gptoss_tiny.gguf", Options{Backend: "cpu", Quant: "int4", DirectLoad: true, ExactPrefill: true,
				Knobs: &Knobs{knobMoEExpertMajor: em}})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			a := m.w.arch
			if a.gptoss == nil || a.SlidingWindow <= 0 {
				t.Fatalf("the fixture must be a windowed gpt-oss: gptoss %v, window %d", a.gptoss != nil, a.SlidingWindow)
			}
			n := 2*a.SlidingWindow + 10
			if !m.canBatchN(n) {
				t.Fatalf("canBatchN(%d) = false: gpt-oss's prompt does not take the batched walker", n)
			}
			_, _, _, _, _, _, vocab := m.Dims()
			prompt := make([]int, n)
			for i := range prompt {
				prompt[i] = 1 + (i*37+11)%(vocab-1)
			}
			const steps = 8
			seq := m.NewCache(n + steps)
			for _, id := range prompt[:n-1] {
				if _, err := m.runLayers(id, seq); err != nil {
					t.Fatal(err)
				}
			}
			want, err := m.forward(prompt[n-1], seq)
			if err != nil {
				t.Fatal(err)
			}
			want = slices.Clone(want)
			bat := m.NewCache(n + steps)
			p0, r0 := atomic.LoadInt64(&prefillPasses), gptOssExpertMajorRuns.Load()
			got, err := m.prefillLogits(context.Background(), prompt, bat)
			if err != nil {
				t.Fatal(err)
			}
			if atomic.LoadInt64(&prefillPasses) == p0 {
				t.Fatal("prefillLogits ran no batched pass")
			}
			if d := gptOssExpertMajorRuns.Load() - r0; (em == "1") != (d > 0) {
				t.Fatalf("expert-major ran %d chunks with the knob at %s", d, em)
			}
			for step := 0; ; step++ {
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("step %d logit %d: batched prefill %v != sequential %v", step, i, got[i], want[i])
					}
				}
				if step == steps {
					break
				}
				next := 0
				for i, v := range want {
					if v > want[next] {
						next = i
					}
				}
				if want, err = m.forward(next, seq); err != nil {
					t.Fatal(err)
				}
				want = slices.Clone(want)
				if got, err = m.forward(next, bat); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

// TestGptOssPrefill_defaultIsExact prefills the tiny gpt-oss at fastAttnMinPrompt rows at the default, where every other batched family
// takes CPU fast attention, with the fast-attention knob explicitly on. gpt-oss's batched prefill must stay on the exact kernel
// (cpuFastAttention false) and its last row's logits must equal the per-token walker's bit for bit.
func TestGptOssPrefill_defaultIsExact(t *testing.T) {
	m, err := Load("testdata/gptoss_tiny.gguf", Options{Backend: "cpu", Quant: "int4", DirectLoad: true, Knobs: &Knobs{knobCPUFastAttention: "1"}})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.cpuFastAttention() {
		t.Fatal("cpuFastAttention() is true for gpt-oss: its batched prefill would take the f32 attention")
	}
	n := fastAttnMinPrompt
	_, _, _, _, _, _, vocab := m.Dims()
	prompt := make([]int, n)
	for i := range prompt {
		prompt[i] = 1 + (i*53+7)%(vocab-1)
	}
	seq := m.NewCache(n + 1)
	for _, id := range prompt[:n-1] {
		if _, err := m.runLayers(id, seq); err != nil {
			t.Fatal(err)
		}
	}
	want, err := m.forward(prompt[n-1], seq)
	if err != nil {
		t.Fatal(err)
	}
	want = slices.Clone(want)
	p0 := atomic.LoadInt64(&prefillPasses)
	got, err := m.prefillLogits(context.Background(), prompt, m.NewCache(n+1))
	if err != nil {
		t.Fatal(err)
	}
	if atomic.LoadInt64(&prefillPasses) == p0 {
		t.Fatal("prefillLogits ran no batched pass")
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("logit %d: batched prefill %v != sequential %v", i, got[i], want[i])
		}
	}
}

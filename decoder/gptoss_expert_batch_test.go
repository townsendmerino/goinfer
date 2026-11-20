package decoder

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
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
	orig := w4a8BatchEnabled
	t.Cleanup(func() { w4a8BatchEnabled = orig })
	capture := func(on bool) ([][]float32, int64) {
		w4a8BatchEnabled = on
		c0 := gptOssExpertBatchCalls.Load()
		var steps [][]float32
		sp := SamplingParams{Temperature: 0, LogitProcessor: func(_ []int, logits []float32) {
			steps = append(steps, append([]float32(nil), logits...))
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

	orig := w4a8BatchEnabled
	t.Cleanup(func() { w4a8BatchEnabled = orig; linalg.SetParallelWidth(0) })
	linalg.SetParallelWidth(8)
	run := func(on bool) ([]float32, int64) {
		w4a8BatchEnabled = on
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

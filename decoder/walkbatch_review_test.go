package decoder

import (
	"context"
	"math"
	"testing"
)

// TestWalkBatch_gatedAttentionIgnoresFastAttention: qwen3.5's gated attention runs with f64 accumulation whatever the request
// asks, so a pass with fastAttn on and a floor it clears must neither build the fused attention pool (whose slots carry no
// score buffers) nor change a bit of the result.
func TestWalkBatch_gatedAttentionIgnoresFastAttention(t *testing.T) {
	m, err := Load("../testdata/qwen35-tiny", Options{})
	if err != nil {
		t.Skipf("no qwen35-tiny fixture: %v", err)
	}
	defer m.Close()
	if !m.program().anyOp(func(o *op) bool { return o.kind == opGatedAttn }) {
		t.Fatal("test bug: qwen35-tiny has no gated-attention layer")
	}
	const n = 24
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*37 + 11) % m.w.arch.VocabSize
	}
	run := func(fast bool, ps *prefillPassScratch) []float32 {
		h, err := m.walkBatch(context.Background(), m.program(), m.embedN(ids), m.NewCache(n), fast, fastAttnMinPrompt, ps, nil)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), h...)
	}
	want := run(false, nil)
	// A scratch whose pool was last built fused, as a reused prefillPassScratch can be.
	stale := &prefillPassScratch{poolWant: true}
	for _, c := range []struct {
		name string
		ps   *prefillPassScratch
	}{{"fresh scratch", nil}, {"scratch with a fused pool", stale}} {
		got := run(true, c.ps)
		for i := range want {
			if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
				t.Fatalf("%s: fastAttn changed row value %d: %v vs %v", c.name, i, got[i], want[i])
			}
		}
	}
}

// TestWalkBatch_gemma4PassTakesNoGenericAttentionScratch: a program whose attention is Gemma 4's own operation (which allocates
// its q, k, v and context per layer) leaves the walker's generic q, k, v and context buffers unallocated.
func TestWalkBatch_gemma4PassTakesNoGenericAttentionScratch(t *testing.T) {
	m, err := Load("../testdata/gemma4-moe-kv-tiny", Options{})
	if err != nil {
		t.Skipf("no gemma4-moe-kv-tiny fixture: %v", err)
	}
	defer m.Close()
	const n = 12
	ids := make([]int, n)
	for i := range ids {
		ids[i] = (i*37 + 11) % m.w.arch.VocabSize
	}
	ps := &prefillPassScratch{}
	if _, err := m.walkBatch(context.Background(), m.program(), m.embedN(ids), m.NewCache(n), false, n, ps, &batchRows{ids: ids}); err != nil {
		t.Fatal(err)
	}
	for name, b := range map[string][]float32{"q": ps.q, "k": ps.k, "v": ps.v, "ctx": ps.ctx, "gate": ps.gate, "up": ps.up} {
		if cap(b) != 0 {
			t.Errorf("Gemma 4 pass allocated the generic %s buffer (%d floats)", name, cap(b))
		}
	}
	if ps.pool != nil {
		t.Errorf("Gemma 4 pass built the generic attention pool")
	}
}

//go:build goinfer_testhooks

package decoder

import (
	"context"
	"testing"
)

// Audit R-12: K=1 decode on an f32 sliding-window ring reads its window in place instead of copying every resident row into scratch each token
// (ringDirectDecode; ring_mirror_test.go gates the bits on tiny checkpoints). This measures it on real Gemma checkpoints with the repo's paired ABBA harness
// (cpuDecodeAB), at a depth past the window so every local layer is full: Gemma 3 1B (window 512) at depth 900 by day, Gemma 2 2B (window 4096) at depth 4500
// on the night queue (its prefill alone is about 2 minutes per generation).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run 'TestCPURing_directDecodeAB/gemma3' -v -count=1 -timeout 30m
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run 'TestCPURing_directDecodeAB/gemma2' -v -count=1 -timeout 60m
func TestCPURing_directDecodeAB(t *testing.T) {
	for _, mc := range []struct {
		name, env, def string
		depth, pairs   int
	}{
		{"gemma3-1b-W512", "GOINFER_CPU_MODEL_G3", "$HOME/models/gemma3-1b-q4_k_m.gguf", 900, 5},
		{"gemma2-2b-W4096", "GOINFER_CPU_MODEL_G2", "$HOME/models/gemma-2-2b-it-Q8_0.gguf", 4500, 3},
	} {
		t.Run(mc.name, func(t *testing.T) {
			h := newCPUDecodeAB(t, mc.env, mc.def, mc.depth)
			orig := ringDirectDecode
			t.Cleanup(func() { ringDirectDecode = orig })
			h.run("ring direct decode ("+mc.name+")", mc.pairs, func(on bool) { ringDirectDecode = on })
		})
	}
}

// TestCPURing_directDecodeLogitsBitIdentical is the strong form of the check on the real Gemma 3 1B past its window: the FULL logits vector at every decode step
// through the sampler's LogitProcessor, in both arms, compared with != on every element (a matching greedy stream proves little about numerics).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestCPURing_directDecodeLogitsBitIdentical -v -count=1 -timeout 30m
func TestCPURing_directDecodeLogitsBitIdentical(t *testing.T) {
	h := newCPUDecodeAB(t, "GOINFER_CPU_MODEL_G3", "$HOME/models/gemma3-1b-q4_k_m.gguf", 900)
	orig := ringDirectDecode
	t.Cleanup(func() { ringDirectDecode = orig })
	capture := func(on bool) [][]float32 {
		ringDirectDecode = on
		var steps [][]float32
		sp := SamplingParams{Temperature: 0, LogitProcessor: func(_ []int, logits []float32) {
			steps = append(steps, append([]float32(nil), logits...))
		}}
		out, g := h.m.Generate(context.Background(), h.ids, 48, sp)
		for range out {
		}
		if g.err != nil {
			t.Fatalf("generate: %v", g.err)
		}
		return steps
	}
	off := capture(false)
	on := capture(true)
	if len(on) != len(off) || len(on) == 0 {
		t.Fatalf("captured %d steps ON vs %d OFF", len(on), len(off))
	}
	bad := 0
	for s := range off {
		for i := range off[s] {
			if on[s][i] != off[s][i] {
				bad++
				if bad <= 3 {
					t.Errorf("step %d logit %d: in-place %v != copy path %v", s, i, on[s][i], off[s][i])
				}
			}
		}
	}
	t.Logf("gemma3-1b depth 900: %d decode steps x %d logits compared, %d differ", len(off), len(off[0]), bad)
}

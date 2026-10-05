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

// Audit R-15: the int8-KV widen is split by rows across goroutines (kvDequantParallel, decoder/kvdequant_parallel.go). Measured on real checkpoints with int8 KV, at a
// depth where the widen is large: Qwen2.5 1.5B (every layer global) at depth 2000, and Gemma 3 1B (int8 rings plus global layers) at depth 900.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestCPUKVI8_parallelWidenAB -v -count=1 -timeout 30m
func TestCPUKVI8_parallelWidenAB(t *testing.T) {
	for _, mc := range []struct {
		name, env, def string
		depth, pairs   int
	}{
		{"qwen2.5-1.5b-depth2000", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 2000, 4},
		{"gemma3-1b-depth900", "GOINFER_CPU_MODEL_G3", "$HOME/models/gemma3-1b-q4_k_m.gguf", 900, 5},
	} {
		t.Run(mc.name, func(t *testing.T) {
			h := newCPUDecodeABWith(t, mc.env, mc.def, mc.depth, Options{KVQuant: "i8"})
			orig := kvDequantParallel
			t.Cleanup(func() { kvDequantParallel = orig })
			before := kvDequantFanouts.Load()
			h.run("int8-KV parallel widen ("+mc.name+")", mc.pairs, func(on bool) { kvDequantParallel = on })
			if kvDequantFanouts.Load() == before {
				t.Fatal("the fan-out never forked: the ON arm was the OFF arm")
			}
		})
	}
}

// TestCPUKVI8_parallelWidenLogitsBitIdentical compares the FULL logits at every decode step, int8 KV, fan-out on and off, on the real Qwen2.5 1.5B at depth 2000.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestCPUKVI8_parallelWidenLogitsBitIdentical -v -count=1 -timeout 30m
func TestCPUKVI8_parallelWidenLogitsBitIdentical(t *testing.T) {
	h := newCPUDecodeABWith(t, "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", 2000, Options{KVQuant: "i8"})
	orig := kvDequantParallel
	t.Cleanup(func() { kvDequantParallel = orig })
	capture := func(on bool) [][]float32 {
		kvDequantParallel = on
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
	before := kvDequantFanouts.Load()
	on := capture(true)
	if kvDequantFanouts.Load() == before {
		t.Fatal("the fan-out never forked")
	}
	if len(on) != len(off) || len(on) == 0 {
		t.Fatalf("captured %d steps ON vs %d OFF", len(on), len(off))
	}
	bad := 0
	for s := range off {
		for i := range off[s] {
			if on[s][i] != off[s][i] {
				bad++
				if bad <= 3 {
					t.Errorf("step %d logit %d: parallel %v != serial %v", s, i, on[s][i], off[s][i])
				}
			}
		}
	}
	t.Logf("qwen2.5-1.5b int8 KV depth 2000: %d decode steps x %d logits compared, %d differ", len(off), len(off[0]), bad)
}

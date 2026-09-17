//go:build gpu && goinfer_testhooks

package gpu_test

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestWebGPU_ForwardNoLogits_byteIdenticalKV verifies that running ForwardNoLogits on tokens 0..N-2
// and Forward on token N-1 writes the exact same resident KV cache as running Forward on all tokens 0..N-1.
// The resulting last-token logits and subsequent autoregressive generation must be 100% bit-identical.
func TestWebGPU_ForwardNoLogits_byteIdenticalKV(t *testing.T) {
	if _, err := gpu.New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}

	fixtures := []string{
		"../testdata/llama-tiny",
		"../testdata/glm-tiny",
	}

	for _, ckpt := range fixtures {
		t.Run(ckpt, func(t *testing.T) {
			if _, err := os.Stat(ckpt); err != nil {
				t.Skipf("checkpoint not found: %s", ckpt)
			}

			m, err := decoder.Load(ckpt, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
			if err != nil {
				t.Fatalf("Load(%s): %v", ckpt, err)
			}
			defer m.Close()

			rf := m.ResidentForwardForTest()
			if rf == nil {
				t.Fatalf("model not resident: %s", m.ResidentDecline())
			}

			kvOnly, ok := rf.(decoder.ResidentPrefillKV)
			if !ok {
				t.Fatalf("residentDecoder does not implement decoder.ResidentPrefillKV")
			}

			prompt := []int{2, 7, 13, 5, 19, 11, 23, 17, 3, 29}

			// 1. Reference: Full-logits Forward for every token
			rf.Reset()
			var wantLogits []float32
			for i, id := range prompt {
				emb := m.EmbedResidentForTest(id)
				l, err := rf.Forward(emb, i)
				if err != nil {
					t.Fatalf("full-logits forward step %d: %v", i, err)
				}
				if i == len(prompt)-1 {
					wantLogits = make([]float32, len(l))
					copy(wantLogits, l)
				}
			}

			// Generate 4 continuation tokens from the reference state
			wantCont := make([]int, 4)
			currEmb := m.EmbedResidentForTest(prompt[len(prompt)-1])
			pos := len(prompt)
			lastL := wantLogits
			for step := 0; step < 4; step++ {
				argmax := 0
				maxVal := float32(-math.MaxFloat32)
				for idx, v := range lastL {
					if v > maxVal {
						maxVal = v
						argmax = idx
					}
				}
				wantCont[step] = argmax
				currEmb = m.EmbedResidentForTest(argmax)
				lastL, err = rf.Forward(currEmb, pos)
				if err != nil {
					t.Fatalf("ref continuation step %d: %v", step, err)
				}
				pos++
			}

			// 2. Test: ForwardNoLogits for tokens 0..N-2, Forward for N-1
			rf.Reset()
			for i := 0; i < len(prompt)-1; i++ {
				emb := m.EmbedResidentForTest(prompt[i])
				if err := kvOnly.ForwardNoLogits(emb, i); err != nil {
					t.Fatalf("ForwardNoLogits step %d: %v", i, err)
				}
			}
			lastEmb := m.EmbedResidentForTest(prompt[len(prompt)-1])
			gotLogits, err := rf.Forward(lastEmb, len(prompt)-1)
			if err != nil {
				t.Fatalf("Forward for last token: %v", err)
			}

			// Compare last-token logits
			if len(wantLogits) != len(gotLogits) {
				t.Fatalf("logits length mismatch: want %d, got %d", len(wantLogits), len(gotLogits))
			}
			var maxDiff float32
			for i := range wantLogits {
				d := float32(math.Abs(float64(wantLogits[i] - gotLogits[i])))
				if d > maxDiff {
					maxDiff = d
				}
				if wantLogits[i] != gotLogits[i] {
					t.Fatalf("last-token logits differ at index %d: want %g, got %g (diff %g)",
						i, wantLogits[i], gotLogits[i], d)
				}
			}
			t.Logf("Last-token logits match bit-identically (maxDiff = %g)", maxDiff)

			// Generate 4 continuation tokens from the KV-only prefilled state
			gotCont := make([]int, 4)
			currEmb = lastEmb
			pos = len(prompt)
			lastL = gotLogits
			for step := 0; step < 4; step++ {
				argmax := 0
				maxVal := float32(-math.MaxFloat32)
				for idx, v := range lastL {
					if v > maxVal {
						maxVal = v
						argmax = idx
					}
				}
				gotCont[step] = argmax
				currEmb = m.EmbedResidentForTest(argmax)
				lastL, err = rf.Forward(currEmb, pos)
				if err != nil {
					t.Fatalf("test continuation step %d: %v", step, err)
				}
				pos++
			}

			for step := range wantCont {
				if wantCont[step] != gotCont[step] {
					t.Fatalf("continuation token %d differs: want %d, got %d", step, wantCont[step], gotCont[step])
				}
			}
			t.Logf("Continuation tokens match bit-identically: %v == %v", wantCont, gotCont)
		})
	}
}

// TestWebGPU_Generate_KVOnlyMatchesFull tests the end-to-end Generate path through decoder.Model,
// asserting that GOINFER_NO_KVONLY_PREFILL="" (enabled) and GOINFER_NO_KVONLY_PREFILL="1" (forced full logits)
// produce the exact same sequence of generated tokens.
func TestWebGPU_Generate_KVOnlyMatchesFull(t *testing.T) {
	if _, err := gpu.New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}

	fixtures := []string{
		"../testdata/llama-tiny",
		"../testdata/glm-tiny",
	}

	for _, ckpt := range fixtures {
		t.Run(ckpt, func(t *testing.T) {
			if _, err := os.Stat(ckpt); err != nil {
				t.Skipf("checkpoint not found: %s", ckpt)
			}

			// Force sequential prefill by disabling batched prefill (so residentPrefillSeed exercises the per-token loop)
			t.Setenv("GOINFER_BATCHED_PREFILL", "0")

			prompt := []int{3, 11, 7, 19, 2, 15, 23, 31, 5, 17, 29}
			const genCount = 16
			greedy := decoder.SamplingParams{Temperature: 0}

			runGen := func(kvOnly bool) []int {
				if kvOnly {
					t.Setenv("GOINFER_NO_KVONLY_PREFILL", "")
				} else {
					t.Setenv("GOINFER_NO_KVONLY_PREFILL", "1")
				}
				m, err := decoder.Load(ckpt, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
				if err != nil {
					t.Fatalf("Load(%s): %v", ckpt, err)
				}
				defer m.Close()

				if _, ok := m.ResidentForwardForTest().(decoder.ResidentPrefillKV); !ok {
					t.Fatal("residentDecoder does not implement decoder.ResidentPrefillKV")
				}

				ch, _ := m.Generate(context.Background(), prompt, genCount, greedy)
				var toks []int
				for tok := range ch {
					toks = append(toks, tok)
				}
				return toks
			}

			on := runGen(true)
			off := runGen(false)

			if len(on) == 0 {
				t.Fatal("no tokens generated with KV-only prefill")
			}
			if len(on) != len(off) {
				t.Fatalf("generated length mismatch: on=%d, off=%d", len(on), len(off))
			}
			for i := range on {
				if on[i] != off[i] {
					t.Fatalf("token %d differs: KV-only=%d, full-logits=%d", i, on[i], off[i])
				}
			}
			t.Logf("Generated %d tokens identical: %v", len(on), on)
		})
	}
}

// BenchmarkWebGPU_Prefill_KVOnlyVsFull benchmarks per-token prefill latency with Forward vs ForwardNoLogits.
func BenchmarkWebGPU_Prefill_KVOnlyVsFull(b *testing.B) {
	if _, err := gpu.New(); err != nil {
		b.Skipf("no WebGPU adapter: %v", err)
	}
	ckpt := "../testdata/llama-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		b.Skipf("checkpoint not found: %s", ckpt)
	}

	m, err := decoder.Load(ckpt, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		b.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	rf := m.ResidentForwardForTest()
	kvOnly, ok := rf.(decoder.ResidentPrefillKV)
	if !ok {
		b.Fatal("not ResidentPrefillKV")
	}

	const promptLen = 16
	embs := make([][]float32, promptLen)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest((i*7 + 3) % 100)
	}

	b.Run("FullLogits_Forward", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rf.Reset()
			for p := 0; p < promptLen; p++ {
				if _, err := rf.Forward(embs[p], p); err != nil {
					b.Fatalf("Forward %d: %v", p, err)
				}
			}
		}
	})

	b.Run("KVOnly_ForwardNoLogits", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rf.Reset()
			for p := 0; p < promptLen-1; p++ {
				if err := kvOnly.ForwardNoLogits(embs[p], p); err != nil {
					b.Fatalf("ForwardNoLogits %d: %v", p, err)
				}
			}
			if _, err := rf.Forward(embs[promptLen-1], promptLen-1); err != nil {
				b.Fatalf("Forward %d: %v", promptLen-1, err)
			}
		}
	})
}

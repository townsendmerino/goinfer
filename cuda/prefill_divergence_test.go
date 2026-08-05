//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillDivergenceRate is the tag-deciding measurement (Task 1): how often does
// batched-prefill-then-decode produce a DIFFERENT greedy token stream than sequential-prefill-then-
// decode, on a real model? The batched prefill writes KV that differs ~2e-6 from what sequential
// decode writes (TestBatchedVsDecodeGap / the real-KV finding); "only bites at rare ties" is an
// inference until measured. This runs many distinct prompts, generates the same length two ways
// (only the prefill differs; the decode loop is identical), and reports how many streams diverge and
// where. Heavy; gated.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags cuda -run TestPrefillDivergenceRate -v -timeout 30m
func TestPrefillDivergenceRate(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	const path = "/home/francis/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	rf := mc.ResidentForwardForTest().(*cudaResident)
	_, _, _, _, _, _, vocab := mc.Dims()
	emb := func(id int) []float32 { return mc.EmbedResidentForTest(((id % vocab) + vocab) % vocab) }

	const (
		prompts   = 50
		promptLen = 48
		genLen    = 128
	)

	// decode genLen tokens from a primed state (lg = last-position logits, pos = next position). The
	// decode loop is IDENTICAL for both arms — only the prime differs — so any divergence is the prefill.
	decodeFrom := func(lg []float32, pos int) []int {
		out := make([]int, 0, genLen)
		cur := lg
		for i := 0; i < genLen; i++ {
			tk := argmaxF(cur)
			out = append(out, tk)
			l, e := rf.Forward(emb(tk), pos)
			if e != nil {
				t.Fatalf("decode: %v", e)
			}
			cur = l
			pos++
		}
		return out
	}

	diverged, firstMin, firstSum := 0, genLen+1, 0
	firsts := []int{}
	var s uint32 = 0x9e3779b9
	for p := 0; p < prompts; p++ {
		prompt := make([]int, promptLen)
		for i := range prompt {
			s = s*1664525 + 1013904223
			prompt[i] = int(s>>9) % vocab
		}
		pe := make([][]float32, promptLen)
		for i, tk := range prompt {
			pe[i] = emb(tk)
		}

		// arm A: BATCHED prefill → decode
		lgA, e := rf.PrefillLast(pe, 0)
		if e != nil {
			t.Fatalf("batched prefill: %v", e)
		}
		streamA := decodeFrom(lgA, promptLen)

		// arm B: SEQUENTIAL prefill → decode
		for i := 0; i < promptLen-1; i++ {
			if e := rf.ForwardNoLogits(pe[i], i); e != nil {
				t.Fatalf("seq prefill: %v", e)
			}
		}
		lgB, e := rf.Forward(pe[promptLen-1], promptLen-1)
		if e != nil {
			t.Fatalf("seq last: %v", e)
		}
		lgB = append([]float32(nil), lgB...)
		streamB := decodeFrom(lgB, promptLen)

		first := -1
		for i := 0; i < genLen; i++ {
			if streamA[i] != streamB[i] {
				first = i
				break
			}
		}
		if first >= 0 {
			diverged++
			firsts = append(firsts, first)
			firstSum += first
			if first < firstMin {
				firstMin = first
			}
		}
	}

	t.Logf("DIVERGENCE RATE (real 1.5B, %d prompts × %d-tok prompt × %d-tok greedy gen):", prompts, promptLen, genLen)
	t.Logf("  streams diverged: %d / %d (%.0f%%)", diverged, prompts, 100*float64(diverged)/float64(prompts))
	if diverged > 0 {
		t.Logf("  first-divergent-token: min=%d mean=%.1f  (of %d)", firstMin, float64(firstSum)/float64(diverged), genLen)
		t.Logf("  first-divergence positions: %v", firsts)
	} else {
		t.Logf("  no divergence observed — the 2e-6 KV gap did not flip any greedy stream in this sample")
	}
}

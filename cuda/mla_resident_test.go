//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMLAResidentParityCUDA verifies that deepseek-tiny (DeepSeek-V2 / V3 architecture with MLA
// compressed-KV latent attention + q-LoRA + ungated shared expert MoE) loads on CUDA, activates
// resident execution with isMLA=true, and produces forward logits matching the CPU reference
// with high cosine similarity across multiple decode positions, as well as greedy generation.
func TestMLAResidentParityCUDA(t *testing.T) {
	const ckpt = "../testdata/deepseek-tiny"
	requireDeviceAndFixture(t, ckpt)

	const quant = "int4"
	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: quant})
	if err != nil {
		t.Fatalf("load cuda (%s): %v", quant, err)
	}
	defer mRes.Close()

	if !mRes.ResidentActive() {
		t.Fatalf("deepseek-tiny did not go CUDA-resident (%s) — decline: %s", quant, mRes.ResidentDecline())
	}

	rf := mRes.ResidentForwardForTest()
	cr, ok := rf.(*cudaResident)
	if !ok {
		t.Fatalf("deepseek-tiny resident forward is %T, want *cudaResident", rf)
	}
	if !cr.isMLA {
		t.Fatalf("deepseek-tiny resident runner has isMLA=false, want true")
	}

	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: quant})
	if err != nil {
		t.Fatalf("load cpu (%s): %v", quant, err)
	}
	defer mCPU.Close()

	prompt := []int{1, 7, 3, 42, 9, 5}

	rf.Reset()
	cache := mCPU.NewCache(len(prompt))
	worstCos := 1.0
	for i, tok := range prompt {
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		cos, _ := cosF32(lr, lc)
		worstCos = math.Min(worstCos, cos)
		t.Logf("[%s] pos %d (tok %d): cosine vs cpu = %.8f", quant, i, tok, cos)
	}
	t.Logf("[%s] MLA resident decode, %d positions: worst cosine %.6f", quant, len(prompt), worstCos)
	if worstCos < 0.99 {
		t.Errorf("[%s] MLA resident forward diverges from CPU: worst cosine %.6f (want >= 0.99)", quant, worstCos)
	}

	// Greedy generation check
	greedy := decoder.SamplingParams{Temperature: 0}
	const N = 8
	gen := func(m *decoder.Model) []int {
		ch, _ := m.Generate(context.Background(), prompt, N, greedy)
		var toks []int
		for id := range ch {
			toks = append(toks, id)
		}
		return toks
	}

	cpuToks := gen(mCPU)
	gpuToks := gen(mRes)
	t.Logf("[%s] MLA generate: cpu=%v cuda=%v", quant, cpuToks, gpuToks)
	if len(cpuToks) == 0 || len(gpuToks) == 0 {
		t.Fatalf("[%s] empty generation: cpu=%v cuda=%v", quant, cpuToks, gpuToks)
	}
	// The FULL sequence, not just the first token: found live reviewing this test — a
	// deliberately-mutated nGroup/topkGroup transposition in the router launch (cuda/resident.go,
	// the trap decoder/features.go's FeatMLA entry names) left the per-position forward-logit
	// cosine check above untouched (worst cosine unchanged) AND left cpuToks[0] == gpuToks[0], so
	// a first-token-only assertion passed clean while tokens 4-7 had already diverged in this same
	// run's own logged output. Discrete expert selection can stay identical for several tokens
	// after a wrong-but-plausible routing decision and only visibly diverge once a different
	// expert combination is actually selected — checking one prefix position is not enough for a
	// selection bug the same way it would be for a smooth numerical one.
	if len(cpuToks) != len(gpuToks) {
		t.Errorf("[%s] MLA resident generation length mismatch: cpu=%d cuda=%d tokens (cpu=%v cuda=%v)",
			quant, len(cpuToks), len(gpuToks), cpuToks, gpuToks)
	} else {
		for i := range cpuToks {
			if cpuToks[i] != gpuToks[i] {
				t.Errorf("[%s] MLA resident generation diverges at token %d: cpu=%d cuda=%d (full: cpu=%v cuda=%v)",
					quant, i, cpuToks[i], gpuToks[i], cpuToks, gpuToks)
				break
			}
		}
	}
}

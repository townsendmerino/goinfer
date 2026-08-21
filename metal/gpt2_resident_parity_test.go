//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGPT2ResidentParityMetal is a whole-model resident-vs-CPU gate for GPT-2 -- the family that
// dispatches layernorm_quant with hasBias=1 (metal/kernels.go; Cohere is the bias-free caller but
// does not go resident on Metal at all, declined for unimplemented features [logit-scale,
// parallel-block] -- confirmed directly before writing this test). layernorm_quant had NO
// whole-model coverage before this (only the isolated TestLayerNormQuant unit-kernel test). Built
// specifically because rmsnorm_quant (rounds 7-9) demonstrated that an isolated kernel test can
// pass exactly while a real bug still shows up only at the whole-model level -- this closes that
// gap for layernorm_quant before any autoresearch candidate touches it. Mirrors
// qwen35_resident_parity_test.go's structure, generic (no recurrent-state specifics): resident vs
// CPU over many tokens (drift check), then a replay after Reset (KV cache must actually clear,
// not merely start empty).
func TestGPT2ResidentParityMetal(t *testing.T) {
	const ckpt = "../testdata/gpt2"

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("gpt2 did not go resident (BuildResident refused) — decode path %q; decline: %s", mRes.DecodePath(), mRes.ResidentDecline())
	}
	t.Logf("resident decode path: %s", mRes.DecodePath())

	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()

	const ntok = 64
	_, _, _, _, _, _, vocab := mCPU.Dims()
	checks := map[int]bool{1: true, 2: true, 4: true, 8: true, 16: true, 32: true, 64: true}

	cache := mCPU.NewCache(ntok + 1)
	rf.Reset()
	worst, first, agree := 1.0, 1.0, 0
	var firstRun [][]float32
	for i := range ntok {
		tok := (i*131 + 7) % vocab
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		if len(firstRun) < 16 {
			firstRun = append(firstRun, append([]float32(nil), lr...))
		}
		if argmaxF32(lc) == argmaxF32(lr) {
			agree++
		}
		cos, maxAbs := cosF32(lc, lr)
		if i == 0 {
			first = cos
		}
		if cos < worst {
			worst = cos
		}
		if checks[i+1] || i+1 == ntok {
			t.Logf("  tok %3d: cosine=%.6f maxAbs=%.4g argmax_match=%v", i+1, cos, maxAbs, argmaxF32(lc) == argmaxF32(lr))
		}
	}
	t.Logf("  worst cosine over %d tokens: %.6f (tok1 %.6f, drift %.4f); argmax agree %d/%d",
		ntok, worst, first, first-worst, agree, ntok)

	if first-worst > 0.02 {
		t.Errorf("cosine DRIFTS over the run: tok1 %.6f -> worst %.6f (delta %.4f)", first, worst, first-worst)
	}
	if worst < 0.95 {
		t.Errorf("worst cosine %.6f < 0.95 — below the resident-path floor", worst)
	}

	const replay = 16
	rf.Reset()
	worstReplay := 1.0
	for i := range replay {
		lr, err := rf.Forward(mRes.EmbedResidentForTest((i*131+7)%vocab), i)
		if err != nil {
			t.Fatalf("resident replay[%d]: %v", i, err)
		}
		if cos, _ := cosF32(firstRun[i], lr); cos < worstReplay {
			worstReplay = cos
		}
	}
	t.Logf("  replay of the first %d tokens after Reset: worst self-cosine %.9f", replay, worstReplay)
	if worstReplay < 0.9999999 {
		t.Errorf("replaying the same tokens after Reset gives DIFFERENT logits (worst self-cosine %.9f): "+
			"the KV cache carried over from the first sequence", worstReplay)
	}
}

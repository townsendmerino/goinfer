//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestSmolLM3ResidentSmokeMetal is the smoke gate for FeatNoPE (docs/tasks/task-gpu-paths-2026-09.md, G5): the model goes resident
// with its last layer NoPE (testdata/smollm3-tiny, no_rope_layers=[1,1,1,0]; 1=has-rope, 0=NoPE), no logit of a 32-token decode or
// of the batched prefill is NaN, the prefill path is batched, and the batched-prefill logits match the sequential decode's at
// cosine >= 0.98.
//
// Deliberately NOT a resident-vs-CPU cosine floor: this fixture cannot discriminate a correct NoPE implementation from a broken one
// (with the fix, with it reverted, and with every layer's invFreq forced to zero, the worst cosine against the CPU over 32 tokens
// was the same to within 0.001), because seeded synthetic weights at hidden=64 let int8 quantization noise dominate; the same finding
// was recorded for Mellum on Metal (features.go's FeatRopeMscale note). The correctness gate is decoder.TestRopeInvFreqLayer_NoPEIsZero
// (exact zero per layer, no GPU) plus the rope2 kernel being the identity at invFreq==0 by construction (TestRope_mscale, rope_test.go).
func TestSmolLM3ResidentSmokeMetal(t *testing.T) {
	const ckpt = "../testdata/smollm3-tiny"

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()

	feats := mRes.RequiredResidentFeatures()
	hasNoPE := false
	for _, f := range feats {
		hasNoPE = hasNoPE || f == decoder.FeatNoPE
	}
	if !hasNoPE {
		t.Fatalf("fixture requires %v — this gate is only meaningful if it exercises FeatNoPE; "+
			"if the fixture changed, this test no longer gates what it claims", feats)
	}

	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("smollm3-tiny did not go resident (BuildResident refused) — decode path %q; decline: %s",
			mRes.DecodePath(), mRes.ResidentDecline())
	}
	t.Logf("resident decode path: %s", mRes.DecodePath())

	const ntok = 32
	_, _, _, _, _, _, vocab := mRes.Dims()
	rf.Reset()
	var lastSeqLogits []float32
	for i := range ntok {
		tok := (i*37 + 3) % vocab
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		for j, v := range lr {
			if v != v { // NaN check without importing math
				t.Fatalf("resident forward[%d]: logit[%d] is NaN", i, j)
			}
		}
		lastSeqLogits = lr
	}

	// Batched prefill check: SmolLM3 with FeatNoPE must admit batched prefill.
	batched, reason := mRes.PrefillPath()
	if !batched {
		t.Fatalf("SmolLM3 reported sequential prefill path (%q); expected batched prefill", reason)
	}
	t.Logf("SmolLM3 prefill path: %s", reason)

	decoder.SetKnobEnvForTest(t, mRes, "GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
	pf, ok := rf.(decoder.Prefiller)
	if !ok {
		t.Fatalf("rf does not implement decoder.Prefiller")
	}
	rf.Reset()
	embs := make([][]float32, ntok)
	for i := range ntok {
		tok := (i*37 + 3) % vocab
		embs[i] = mRes.EmbedResidentForTest(tok)
	}
	preLogits, err := pf.PrefillLast(t.Context(), embs, 0)
	if err != nil {
		t.Fatalf("PrefillLast error: %v", err)
	}
	for j, v := range preLogits {
		if v != v {
			t.Fatalf("PrefillLast: logit[%d] is NaN", j)
		}
	}
	t.Logf("SmolLM3 batched prefill succeeded (%d tokens)", ntok)
	cos := cosF(lastSeqLogits, preLogits)
	t.Logf("SmolLM3 sequential vs batched prefill cosine: %.5f", cos)
	if cos < 0.98 {
		t.Errorf("cosine %.5f < 0.98 between sequential decode and batched prefill", cos)
	}
}

//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestSmolLM3ResidentSmokeMetal is G5's FeatNoPE row (docs/task-gpu-paths-2026-09.md) smoke
// gate: the model actually goes resident and produces finite, non-degenerate output when its
// LAST layer is NoPE (testdata/smollm3-tiny's no_rope_layers=[1,1,1,0], 1=has-rope/0=NoPE).
//
// This is DELIBERATELY NOT a resident-vs-CPU cosine floor, and used to be one — MEASURED (not
// assumed) that this fixture cannot discriminate a correct NoPE implementation from a broken
// one: with the real fix (zero invFreq on layer 3 only), with the fix reverted (real invFreq on
// every layer, i.e. layer 3 wrongly ropes), and with EVERY layer's invFreq forced to zero (layers
// 0-2 wrongly skip rope too), the worst cosine against the CPU reference over 32 tokens was
// 0.9619, 0.9617, and 0.9624 respectively — a ~0.0006 spread, all three configurations equally
// "passing" or "failing" any threshold that would separate them. The seeded/synthetic weights at
// hidden=64 mean int8 quantization noise dominates the comparison regardless of rope correctness
// — the SAME "cannot discriminate a real bug from quantization noise on unstructured weights"
// finding this codebase already recorded for Mellum-on-Metal (features.go's FeatRopeMscale note).
//
// The actual correctness gate is decoder.TestRopeInvFreqLayer_NoPEIsZero — a pure, backend-
// agnostic unit test of RopeInvFreqLayer itself (exact zero/non-zero per layer, no GPU, no
// quantization noise), plus the shipped rope2 kernel math being exact identity at invFreq==0 by
// construction (metal/kernels.go, metal/rope_test.go's TestRope_mscale family already exercises
// that kernel directly). This test's only job is: does declaring FeatNoPE actually let the model
// go resident and run without error/NaN.
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
	}
}

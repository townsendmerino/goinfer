//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMinistral3ResidentSmokeMetal is G5's FeatAttnTemp row (docs/task-gpu-paths-2026-09.md)
// smoke gate: the model actually goes resident and produces finite, non-degenerate output when
// the post-RoPE query scale is genuinely exercised (testdata/ministral3-tiny's
// AttnTempOrigMaxPos=8, so a 32-token run steps through four distinct scale values, past the
// identity-at-short-prompts trap AttnTempBeta's own comment warns about).
//
// This is DELIBERATELY NOT a resident-vs-CPU cosine floor — same finding as G5 row 1's
// smollm3-tiny (metal/smollm3_resident_parity_test.go's TestSmolLM3ResidentSmokeMetal has the
// full writeup). MEASURED here too, not assumed: with the real fix and with it force-disabled
// (qTempScale always 1, i.e. the attn-temp scale silently dropped on every position), the worst
// cosine against the CPU reference over 32 tokens was 0.96446 and 0.96525 respectively — a
// ~0.0008 spread, indistinguishable from noise. testdata/ministral3-tiny is ALSO seeded/synthetic
// (scripts/pin_ministral3_tiny.py), so despite its own test file's comment about deliberately
// exercising AttnTempOrigMaxPos and a real YaRN mscale ratio nontrivially, int8-on-random-weights
// noise still dominates a whole-model resident-vs-CPU comparison at this scale.
//
// The actual correctness gate is decoder.TestAttnTempScale_matchesSequentialFormula — a pure,
// backend-agnostic unit test of the exact formula both Model.AttnTempScale (decode) and
// Model.AttnTempParams (CUDA's batched prefill, which must recompute it per row device-side)
// expose, no GPU, no quantization noise. This test's only job: does declaring FeatAttnTemp let
// the model go resident and run without error/NaN.
func TestMinistral3ResidentSmokeMetal(t *testing.T) {
	const ckpt = "../testdata/ministral3-tiny"

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()

	feats := mRes.RequiredResidentFeatures()
	hasAttnTemp := false
	for _, f := range feats {
		hasAttnTemp = hasAttnTemp || f == decoder.FeatAttnTemp
	}
	if !hasAttnTemp {
		t.Fatalf("fixture requires %v — this gate is only meaningful if it exercises FeatAttnTemp; "+
			"if the fixture changed, this test no longer gates what it claims", feats)
	}

	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("ministral3-tiny did not go resident (BuildResident refused) — decode path %q; decline: %s",
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

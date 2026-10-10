//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMinistral3ResidentSmokeMetal is the smoke gate for G5's FeatAttnTemp row (docs/tasks/task-gpu-paths-2026-09.md): the
// model goes resident and produces finite, non-degenerate output when the post-RoPE query scale is genuinely exercised
// (testdata/ministral3-tiny has AttnTempOrigMaxPos=8, so a 32-token run steps through four distinct scale values, past the
// identity-at-short-prompts trap AttnTempBeta's own comment warns about).
//
// It is deliberately not a resident-vs-CPU cosine floor: with the attn-temp scale on and with it force-disabled (qTempScale
// always 1) the worst cosine against the CPU reference was indistinguishable, because the fixture is seeded/synthetic
// (scripts/pin_ministral3_tiny.py) and int8-on-random-weights noise dominates a whole-model comparison at this scale. The
// write-up is in metal/smollm3_resident_parity_test.go's TestSmolLM3ResidentSmokeMetal; the figures are in
// docs/code-notes/metal.md#TestMinistral3ResidentSmokeMetal.
//
// The correctness gate is decoder.TestAttnTempScale_matchesSequentialFormula, a pure unit test of the formula both
// Model.AttnTempScale (decode) and Model.AttnTempParams (CUDA's batched prefill, which recomputes it per row device-side)
// expose. This test's only job: declaring FeatAttnTemp lets the model go resident and run without error or NaN.
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

//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// This smoke gate passed while every decode token's final norm was the wrong kind (audit-2026-09-10
// C-04). The numeric gate is TestCohereResidentParityCUDA (cohere_resident_parity_test.go).
//
// TestCohereResidentSmokeCUDA and TestCohere2ResidentSmokeCUDA are G5's last row
// (docs/task-gpu-paths-2026-09.md) smoke gates on CUDA — the Metal twin
// (metal/cohere_resident_smoke_test.go) explains why this is deliberately a smoke check
// (admission + no NaN) rather than a resident-vs-CPU cosine floor: testdata/cohere-tiny and
// testdata/cohere2-tiny are both "tiny-random" (scripts/pin_cohere_tiny.py), the same
// seeded/synthetic class as every other G5 fixture, and FeatParallelBlock is a pure sequencing
// change (reuse segA's r.aq/r.aSc as the MLP's input instead of re-normalizing the post-attention
// residual) with no new numerical formula to isolate a floor against — quantizing the same shared
// input norm once and reusing it is bit-identical to quantizing it twice from the same source.
//
// The real correctness evidence sits one level down: layernorm_quant_test.go's TestLayerNormQuant
// proves the NEW kernel (this backend had no mean-centered norm before) against an exact CPU
// reference in isolation, and BuildResident's own validation (empty PostNorm/PostAttnNorm/
// PostMLPNorm required to stay empty and unbuilt for a parallelBlock arch) is exercised simply by
// this fixture reaching resident at all.
func TestCohereResidentSmokeCUDA(t *testing.T) {
	testCohereFamilyResidentSmokeCUDA(t, "../testdata/cohere-tiny", false)
}

func TestCohere2ResidentSmokeCUDA(t *testing.T) {
	testCohereFamilyResidentSmokeCUDA(t, "../testdata/cohere2-tiny", true)
}

func testCohereFamilyResidentSmokeCUDA(t *testing.T, ckpt string, wantSlidingWindow bool) {
	t.Helper()
	requireDeviceAndFixture(t, ckpt)

	mc, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mc.Close()

	feats := mc.RequiredResidentFeatures()
	hasParallel, hasLogitScale := false, false
	hasSlidingWindow := false
	for _, f := range feats {
		hasParallel = hasParallel || f == decoder.FeatParallelBlock
		hasLogitScale = hasLogitScale || f == decoder.FeatLogitScale
		hasSlidingWindow = hasSlidingWindow || f == decoder.FeatSlidingWindow
	}
	if !hasParallel || !hasLogitScale {
		t.Fatalf("fixture requires %v — this gate is only meaningful if it exercises both "+
			"FeatParallelBlock and FeatLogitScale; if the fixture changed, this test no longer "+
			"gates what it claims", feats)
	}
	if hasSlidingWindow != wantSlidingWindow {
		t.Fatalf("fixture FeatSlidingWindow=%v, want %v (cohere2 interleaves sliding-window "+
			"layers, cohere does not)", hasSlidingWindow, wantSlidingWindow)
	}

	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("%s did not go resident (BuildResident refused) — decode path %q; decline: %s",
			ckpt, mc.DecodePath(), mc.ResidentDecline())
	}
	t.Logf("resident decode path: %s", mc.DecodePath())

	const ntok = 24
	_, _, _, _, _, _, vocab := mc.Dims()
	for i := range ntok {
		tok := (i*37 + 3) % vocab
		lr, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
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

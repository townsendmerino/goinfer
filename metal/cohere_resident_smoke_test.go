//go:build darwin && goinfer_testhooks

package metal

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestCohereResidentSmokeMetal and TestCohere2ResidentSmokeMetal are G5's last row
// (docs/tasks/task-gpu-paths-2026-09.md) smoke gates: FeatParallelBlock + FeatLogitScale (+
// FeatLayerNorm, already declared here for GPT-2) — Cohere/Command-R and Cohere2/Command-R7B
// going resident instead of running fully on CPU.
//
// DELIBERATELY NOT a resident-vs-CPU cosine floor — same finding every G5 row's tiny fixture has
// recorded (testdata/cohere-tiny / testdata/cohere2-tiny are both "tiny-random" per
// scripts/pin_cohere_tiny.py, the same seeded/synthetic class as olmo3-tiny/ministral3-tiny).
// The real correctness evidence sits one level down, at two INDEPENDENT places:
//
//   - Metal's layernorm_quant kernel already has its own isolated, exact-CPU-reference gate
//     (gpt2_kernels_test.go's TestLayerNormQuant) that explicitly exercises the bias-free branch
//     "Cohere's hasBias=0 path" — this predates this row entirely; wiring Cohere in changes
//     nothing about that kernel or its coverage.
//   - FeatParallelBlock is a pure SEQUENCING change (reuse encodeAttention's r.aq/r.aSc as the
//     MLP's input instead of re-normalizing post-attention x), not a new numerical formula: since
//     layernorm_quant's output is a deterministic function of its input with no per-call
//     randomness, quantizing the shared input norm ONCE and reusing it is bit-identical to
//     quantizing it twice from the same source — there is no precision difference to measure, only
//     a wiring-correctness question (right buffers, right order, right skip conditions), which is
//     exactly what BuildResident's admission validation (empty PostNorm/PostAttnNorm/PostMLPNorm
//     required to stay empty and unbuilt) and this smoke test's admission + no-NaN check cover.
func TestCohereResidentSmokeMetal(t *testing.T) {
	testCohereFamilyResidentSmoke(t, "../testdata/cohere-tiny", false)
}

func TestCohere2ResidentSmokeMetal(t *testing.T) {
	testCohereFamilyResidentSmoke(t, "../testdata/cohere2-tiny", true)
}

func testCohereFamilyResidentSmoke(t *testing.T, ckpt string, wantSlidingWindow bool) {
	t.Helper()
	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()

	feats := mRes.RequiredResidentFeatures()
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
			"layers, cohere does not) — if this changed, olmo_hybrid's sibling gate may need the "+
			"same check", hasSlidingWindow, wantSlidingWindow)
	}

	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		// CORRECTION 2026-10-01 (written on the CUDA box, NOT run on a Mac): Cohere/Cohere2 rotate
		// GPT-J PAIRWISE (dims 2d, 2d+1) and Metal's `rope` kernel is NeoX half-split, so
		// FeatPairwiseRoPE is declared by CUDA only and Metal now DECLINES these families to the CPU path.
		// This smoke test could not see why that matters: it checks admission + no NaN, and a wrong
		// rotation is exact at position 0 and fluent-looking after (the CUDA resident ran real
		// Command-R7B / Aya wrong that way, worst per-position cosine -0.075 / -0.041). Until Metal has
		// pairwise rope kernels, a decline naming the feature is the PASS; when it declares the
		// feature this test falls through to the original admission + no-NaN smoke below, and the
		// session that ports the kernels must add a peaked-attention resident-vs-CPU gate beside it
		// (cuda/pairwise_rope_resident_parity_test.go is the template).
		if why := mRes.ResidentDecline(); !strings.Contains(why, string(decoder.FeatPairwiseRoPE)) {
			t.Fatalf("%s did not go resident and the decline does not name %q (decode path %q; decline: %s)",
				ckpt, decoder.FeatPairwiseRoPE, mRes.DecodePath(), why)
		}
		t.Logf("declined as intended until Metal has pairwise rope kernels: %s", mRes.ResidentDecline())
		return
	}
	t.Logf("resident decode path: %s", mRes.DecodePath())

	const ntok = 24
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

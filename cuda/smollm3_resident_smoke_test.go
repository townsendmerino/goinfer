//go:build cuda && goinfer_testhooks

package cuda

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestSmolLM3ResidentSmokeCUDA is G5's FeatNoPE row (docs/task-gpu-paths-2026-09.md) smoke gate
// on CUDA — the Metal twin of this test (metal/smollm3_resident_parity_test.go) explains at
// length why this is deliberately NOT a resident-vs-CPU cosine floor: on testdata/smollm3-tiny's
// seeded/synthetic weights (hidden=64), a correct NoPE implementation, a reverted one (every
// layer wrongly ropes), and a maximally-broken one (every layer wrongly skips rope) all land
// within ~0.0006 cosine of each other against the CPU reference — int8-on-random-weights noise
// dominates regardless of rope correctness, the same finding this codebase already recorded for
// Mellum-on-Metal. The real correctness gate is decoder.TestRopeInvFreqLayer_NoPEIsZero (a pure
// unit test of the actual fix, no GPU, no quantization noise) plus the shipped rope kernel math
// being exact identity at invFreq==0 by construction. This test's only job: does declaring
// FeatNoPE let the model go resident on CUDA and run without error/NaN.
//
// WRITTEN, NOT RUN: no CUDA device was available while writing this (see
// docs/task-gpu-paths-2026-09.md's G5 status log) — needs a real run on a CUDA box before it can
// be trusted, same posture as G4's cudaResident.HiddenLast.
func TestSmolLM3ResidentSmokeCUDA(t *testing.T) {
	const ckpt = "../testdata/smollm3-tiny"
	requireDeviceAndFixture(t, ckpt)

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
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

//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestOlmo3ResidentSmokeMetal and TestOlmoHybridResidentSmokeMetal are the smoke gates for FeatPostOnlyNorm + FeatQKNormWhole
// (docs/tasks/task-gpu-paths-2026-09.md, G5): the model goes resident, no logit of a 24-token decode (or, for Olmo 3, of the
// batched prefill) is NaN, and the prefill path is as expected (Olmo 3 batched; Olmo Hybrid declines it for its DeltaNet
// recurrence).
//
// Deliberately NOT a resident-vs-CPU cosine floor: the seeded synthetic fixtures (testdata/olmo3-tiny, testdata/olmo_hybrid-tiny)
// cannot separate a real bug from quantization noise. The correctness gate is one level lower: TestQKNorm_wholeVector
// (qknorm_whole_test.go) compares the whole-vector qk_norm dispatch geometry against the production kernel with an exact
// per-component comparison. The post-only pre-norm skip (quant_vec instead of rmsnorm_quant) reuses a kernel ctx-before-o-proj
// already dispatches, so it rests on that coverage plus the admission and no-NaN checks here.
func TestOlmo3ResidentSmokeMetal(t *testing.T) {
	testOlmoFamilyResidentSmoke(t, "../testdata/olmo3-tiny")
}

func TestOlmoHybridResidentSmokeMetal(t *testing.T) {
	testOlmoFamilyResidentSmoke(t, "../testdata/olmo_hybrid-tiny")
}

func testOlmoFamilyResidentSmoke(t *testing.T, ckpt string) {
	t.Helper()
	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load metal: %v", err)
	}
	defer mRes.Close()

	feats := mRes.RequiredResidentFeatures()
	hasPostOnly, hasQKWhole := false, false
	for _, f := range feats {
		hasPostOnly = hasPostOnly || f == decoder.FeatPostOnlyNorm
		hasQKWhole = hasQKWhole || f == decoder.FeatQKNormWhole
	}
	if !hasPostOnly || !hasQKWhole {
		t.Fatalf("fixture requires %v — this gate is only meaningful if it exercises both "+
			"FeatPostOnlyNorm and FeatQKNormWhole; if the fixture changed, this test no longer "+
			"gates what it claims", feats)
	}
	t.Logf("required feats: %v, missing from prefill: %v", feats, mRes.MissingResidentFeatures(prefillFeatures))

	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("%s did not go resident (BuildResident refused) — decode path %q; decline: %s",
			ckpt, mRes.DecodePath(), mRes.ResidentDecline())
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

	// Prefill check: Olmo 3 admits batched prefill; Olmo Hybrid declines due to DeltaNet recurrence.
	mr, ok := rf.(*metalResident)
	if !ok {
		t.Fatalf("rf is not *metalResident")
	}
	batched, reason := mRes.PrefillPath()
	if mr.r.dnet != nil {
		if batched {
			t.Fatalf("Olmo Hybrid has recurrent DeltaNet state and must decline batched prefill, got %s", reason)
		}
		t.Logf("Olmo Hybrid properly declined batched prefill: %s", reason)
	} else {
		if !batched {
			t.Fatalf("Olmo 3 reported sequential prefill path (%q); expected batched prefill", reason)
		}
		t.Logf("Olmo 3 prefill path: %s", reason)

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
			t.Fatalf("Olmo 3 PrefillLast error: %v", err)
		}
		for j, v := range preLogits {
			if v != v {
				t.Fatalf("Olmo 3 PrefillLast: logit[%d] is NaN", j)
			}
		}
		t.Logf("Olmo 3 batched prefill succeeded with %d tokens", ntok)
	}
}

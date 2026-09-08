//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestOlmo3ResidentSmokeMetal and TestOlmoHybridResidentSmokeMetal are G5's FeatPostOnlyNorm +
// FeatQKNormWhole row (docs/task-gpu-paths-2026-09.md) smoke gates: the model actually goes
// resident and produces finite, non-degenerate output.
//
// DELIBERATELY NOT a resident-vs-CPU cosine floor — same finding G5 rows 1-2 already recorded for
// their own seeded/synthetic tiny fixtures (testdata/olmo3-tiny, testdata/olmo_hybrid-tiny are
// both "seeded" per their own pin_*.py scripts, same pattern). This feature has no new pure-Go
// formula the way FeatNoPE/FeatAttnTemp did, so the real correctness gate is one level lower:
// TestQKNorm_wholeVector (qknorm_whole_test.go) proves the whole-vector qk_norm DISPATCH GEOMETRY
// directly against the real production kernel with an exact per-component comparison (no GPU
// quantization noise at all in that path) — the strongest evidence this row has. The postOnly
// pre-norm skip (quant_vec instead of rmsnorm_quant) reuses an already-proven, unmodified kernel
// (ctx-before-o-proj already dispatches it), so its own correctness rests on that kernel's
// existing coverage plus this smoke test's admission-and-no-NaN check.
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
}

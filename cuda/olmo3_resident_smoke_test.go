//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestOlmo3ResidentSmokeCUDA and TestOlmoHybridResidentSmokeCUDA are G5's FeatPostOnlyNorm +
// FeatQKNormWhole row (docs/tasks/task-gpu-paths-2026-09.md) smoke gates on CUDA — the Metal twin
// (metal/olmo3_resident_smoke_test.go) explains why this is deliberately a smoke check (admission
// + no NaN) rather than a resident-vs-CPU cosine floor: testdata/olmo3-tiny and
// testdata/olmo_hybrid-tiny are both seeded/synthetic, and this feature has no new pure-Go
// formula to unit-test the way FeatNoPE/FeatAttnTemp did — the real correctness evidence is
// metal/qknorm_whole_test.go's isolated kernel-geometry proof (exact per-component comparison,
// no GPU quantization noise) plus the fact that CUDA's postOnly/qkNormWhole changes reuse
// ALREADY-SHIPPED, unmodified kernels (quant_vec, qk_norm) with different Go-side launch
// arguments — no new .cu code, no PTX regeneration, unlike G5 row 2 (Ministral 3).
//
// Olmo Hybrid additionally gates a bug fix: both backends assumed every qwen35Params-carrying
// family's full-attention layer used qwen3.5's own double-width q-gate scheme
// (Qwen35ResidentParams hardcoded attnGate=true), while Olmo Hybrid's is plain (olmo3's own scheme);
// Architecture.qwen35.AttnGate now carries it. This smoke test is the regression gate for that fix:
// without it BuildResident declines ("qwen35 softmax layer has empty q_norm/k_norm").
func TestOlmo3ResidentSmokeCUDA(t *testing.T) {
	testOlmoFamilyResidentSmokeCUDA(t, "../testdata/olmo3-tiny")
}

func TestOlmoHybridResidentSmokeCUDA(t *testing.T) {
	testOlmoFamilyResidentSmokeCUDA(t, "../testdata/olmo_hybrid-tiny")
}

func testOlmoFamilyResidentSmokeCUDA(t *testing.T, ckpt string) {
	t.Helper()
	requireDeviceAndFixture(t, ckpt)

	mc, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mc.Close()

	feats := mc.RequiredResidentFeatures()
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

	// Prefill check: both admit batched prefill. Olmo Hybrid's DeltaNet layers take prefillDeltaNetRows
	// (docs/tasks/task-cuda-deltanet-prefill-2026-09.md); at 24 rows the projections run the batched GEMM, so the bar
	// against the per-token pass above is that task's: argmax equal, cosine >= 0.9999.
	cr, isCR := rf.(*cudaResident)
	if isCR {
		embs := make([][]float32, ntok)
		for i := range ntok {
			tok := (i*37 + 3) % vocab
			embs[i] = mc.EmbedResidentForTest(tok)
		}
		rf.Reset()
		var ref []float32
		for i, e := range embs {
			l, err := rf.Forward(e, i)
			if err != nil {
				t.Fatalf("per-token reference forward[%d]: %v", i, err)
			}
			ref = append(ref[:0], l...)
		}
		rf.Reset()
		got, err := cr.PrefillLast(context.Background(), embs, 0)
		if err != nil {
			t.Fatalf("PrefillLast: %v (batched prefill should not decline; recurrent=%v)", err, cr.dnet != nil)
		}
		argmax := func(v []float32) int {
			b := 0
			for i := range v {
				if v[i] > v[b] {
					b = i
				}
			}
			return b
		}
		var ab, aa, bb float64
		for j := range got {
			if got[j] != got[j] {
				t.Fatalf("PrefillLast: logit[%d] is NaN", j)
			}
			ab += float64(ref[j]) * float64(got[j])
			aa += float64(ref[j]) * float64(ref[j])
			bb += float64(got[j]) * float64(got[j])
		}
		cs := ab / math.Sqrt(aa*bb)
		if argmax(ref) != argmax(got) || cs < 0.9999 {
			t.Fatalf("batched prefill vs per-token: argmax %d vs %d, cosine %.7f (bar: equal, >= 0.9999)",
				argmax(got), argmax(ref), cs)
		}
		t.Logf("batched prefill of %d tokens matches per-token (recurrent=%v): argmax %d, cosine %.7f",
			ntok, cr.dnet != nil, argmax(got), cs)
	}
}

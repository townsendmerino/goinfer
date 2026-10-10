//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMinistral3ResidentParityCUDA is the end-to-end gate for G5's FeatAttnTemp row
// (docs/tasks/task-gpu-paths-2026-09.md) on CUDA: declaring FeatAttnTemp is real machinery (a new
// qTempScale parameter threaded through rope_kv/rope_kv_batched, decoder.Model.AttnTempScale/
// AttnTempParams), not just an admission unlock. testdata/ministral3-tiny's AttnTempOrigMaxPos=8
// means even a modest token count steps through several distinct scale values.
//
// HONEST LIMIT: unlike G5 row 1's smollm3-tiny, this cosine floor does NOT discriminate the fix from
// a disabled one (qTempScale always 1): the worst cosine over 32 tokens against the CPU reference is
// indistinguishable between the two. That is not noise-dominance (CUDA's int4 path is far tighter than
// Metal's int8) but EFFECT SIZE: probing the two configurations' own resident logits directly shows
// positions 0-7 (floor=0, scale exactly 1 either way) bit-identical and positions 8+ (floor>0, scale
// ~1.06-1.28) genuinely differing — real and reproducible, but too small relative to this tiny seeded
// model's own output variance for a whole-model cosine floor to isolate. What the body asserts is
// that the feature does not CORRUPT anything: no NaN and worst cosine >= 0.98 (argmax agreement is
// logged, not asserted). The feature-specific correctness proof is
// decoder.TestAttnTempScale_matchesSequentialFormula (the exact formula, no GPU, no quantization
// noise) plus the reviewed kernel math (gemv_fwd.cu/prefill_batched.cu's own comments).
func TestMinistral3ResidentParityCUDA(t *testing.T) {
	const ckpt = "../testdata/ministral3-tiny"
	requireDeviceAndFixture(t, ckpt)

	mc, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mc.Close()

	feats := mc.RequiredResidentFeatures()
	hasAttnTemp := false
	for _, f := range feats {
		hasAttnTemp = hasAttnTemp || f == decoder.FeatAttnTemp
	}
	if !hasAttnTemp {
		t.Fatalf("fixture requires %v — this gate is only meaningful if it exercises FeatAttnTemp; "+
			"if the fixture changed, this test no longer gates what it claims", feats)
	}

	rf := mc.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("ministral3-tiny did not go resident (BuildResident refused) — decode path %q; decline: %s",
			mc.DecodePath(), mc.ResidentDecline())
	}
	t.Logf("resident decode path: %s", mc.DecodePath())

	mcpu, err := decoder.Load(ckpt, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mcpu.Close()

	const ntok = 32
	_, _, _, _, _, _, vocab := mcpu.Dims()
	cache := mcpu.NewCache(ntok + 1)
	worst, first := 1.0, 1.0
	exact := 0
	for i := range ntok {
		tok := (i*37 + 3) % vocab
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		gpuL, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("cuda pos %d: %v", i, err)
		}
		var dot, na, nb float64
		for j := range cpuL {
			dot += float64(cpuL[j]) * float64(gpuL[j])
			na += float64(cpuL[j]) * float64(cpuL[j])
			nb += float64(gpuL[j]) * float64(gpuL[j])
		}
		c := dot / (math.Sqrt(na) * math.Sqrt(nb))
		if math.IsNaN(c) {
			t.Fatalf("pos %d (floor=%d): logit cosine is NaN — degenerate resident output", i, i/8)
		}
		if i == 0 {
			first = c
		}
		if c < worst {
			worst = c
		}
		if argmaxF(cpuL) == argmaxF(gpuL) {
			exact++
		}
		t.Logf("  tok %2d (floor=%d): cosine=%.6f argmax_match=%v", i, i/8, c, argmaxF(cpuL) == argmaxF(gpuL))
	}
	t.Logf("Ministral3 resident vs CPU: %d/%d exact | worst cosine %.6f (tok0 %.6f, drift %.4f)",
		exact, ntok, worst, first, first-worst)

	if worst < 0.98 {
		t.Errorf("worst cosine %.6f < 0.98 — below the resident-path floor for an attn-temp-carrying model", worst)
	}
}

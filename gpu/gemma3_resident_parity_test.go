//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// gemma3ParityPrompt mirrors metal/gemma4_twogeom_test.go's twoGeomPrompt — an arbitrary fixed
// token sequence, not tied to the fixture's real tokenizer.
var gemma3ParityPrompt = []int{1, 7, 42, 100, 5, 200, 13, 88}

// TestGemma3ResidentParityWebGPU is G6's Gemma set gate (docs/tasks/task-gpu-paths-2026-09.md):
// FeatEmbedScale + FeatSandwichNorm + FeatGatedGELU (plus the already-declared
// FeatQKNorm/FeatSlidingWindow/FeatRMSAddOne) against a real, non-seeded checkpoint
// (testdata/gemma3-vl-tiny's text tower, not a tiny-random fixture), so the floor is numeric
// rather than admission-and-no-NaN: resident vs CPU, both int4, at every position.
//
// FeatFinalLogitSoftcap is NOT exercised here (this fixture has no final_logit_softcapping); its
// correctness rests on TestApplySoftcap_bitIdentical (softcap_test.go), a direct port of
// cuda/metal's own gate, since gpu/softcap.go is a byte-identical copy of their applySoftcap.
//
// Gemma 4's dense-twogeom/dense-scaled fixtures are not used here: their per-layer head_dim
// differs between local and global attention layers, and they are gated by
// TestGemma4DenseTwoGeom_residentParity and TestGemma4DenseScaled_webgpuParity
// (gemma4_twogeom_test.go).
func TestGemma3ResidentParityWebGPU(t *testing.T) {
	dir := "../testdata/gemma3-vl-tiny"
	// Stat the WEIGHTS, not the directory: the dir and its config.json are tracked while the weights
	// are gitignored, so a dir stat passes on every clone and the test would then skip saying "no
	// webgpu device" when Load failed on the missing weights.
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no fixture weights (%s/model.safetensors; config.json alone is tracked)", dir)
	}
	if c, err := New(); err != nil {
		t.Skipf("no webgpu device: %v", err)
	} else {
		c.Close()
	}
	if !decoder.ResidentBackendFeatures("webgpu")[decoder.FeatSandwichNorm] {
		t.Skip("webgpu does not declare FeatSandwichNorm yet")
	}

	mg, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("%s did not go resident (BuildResident refused) — decode path %q; decline: %s",
			dir, mg.DecodePath(), mg.ResidentDecline())
	}
	feats := mg.RequiredResidentFeatures()
	hasEmbedScale, hasSandwich, hasGeglu := false, false, false
	for _, f := range feats {
		hasEmbedScale = hasEmbedScale || f == decoder.FeatEmbedScale
		hasSandwich = hasSandwich || f == decoder.FeatSandwichNorm
		hasGeglu = hasGeglu || f == decoder.FeatGatedGELU
	}
	if !hasEmbedScale || !hasSandwich || !hasGeglu {
		t.Fatalf("fixture requires %v — this gate is only meaningful if it exercises "+
			"FeatEmbedScale, FeatSandwichNorm and FeatGatedGELU; if the fixture changed, this "+
			"test no longer gates what it claims", feats)
	}
	t.Logf("resident decode path: %s", mg.DecodePath())

	mcpu, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mcpu.Close()

	cache := mcpu.NewCache(len(gemma3ParityPrompt))
	minCos := 1.0
	for i, tok := range gemma3ParityPrompt {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		gpuL, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward pos %d: %v", i, err)
		}
		var dot, na, nb float64
		for j := range cpuL {
			v, w := float64(cpuL[j]), float64(gpuL[j])
			if v != v || w != w { // NaN check without importing math twice
				t.Fatalf("pos %d: NaN in logits (cpu=%v gpu=%v at %d)", i, v, w, j)
			}
			dot += v * w
			na += v * v
			nb += w * w
		}
		cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
		if cos < minCos {
			minCos = cos
		}
		t.Logf("  pos %2d cosine %.6f", i, cos)
	}
	t.Logf("gemma3-vl-tiny text tower, resident vs CPU (int4 both sides): minCosine=%.6f", minCos)
	// A real numeric floor, not a smoke-test bar: 0.999, against a genuine checkpoint rather than a
	// seeded or synthetic one.
	if minCos < 0.999 {
		t.Errorf("minCosine %.6f < 0.999 — resident diverges from CPU on a real checkpoint", minCos)
	}
}

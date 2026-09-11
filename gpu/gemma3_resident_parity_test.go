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

// TestGemma3ResidentParityWebGPU is G6's Gemma set gate (docs/task-gpu-paths-2026-09.md):
// FeatEmbedScale + FeatSandwichNorm + FeatGatedGELU (+ the already-declared FeatQKNorm/
// FeatSlidingWindow/FeatRMSAddOne) against a REAL, non-seeded checkpoint
// (testdata/gemma3-vl-tiny's text tower — a real small Gemma3 VL model, not a tiny-random
// fixture), unlike every G5 smoke test this session wrote. That makes this a genuine numeric
// floor, not just an admission-and-no-NaN check: resident vs CPU, both int4, at every position.
//
// FeatFinalLogitSoftcap is NOT exercised here (this fixture has no final_logit_softcapping) —
// its own correctness rests on TestApplySoftcap_bitIdentical (softcap_test.go), a direct port of
// cuda/metal's own gate, since gpu/softcap.go is a byte-identical copy of their applySoftcap.
//
// Gemma 4's dense-twogeom/dense-scaled fixtures are NOT used here: both have a per-layer
// head_dim that differs between local and global attention layers (head_dim 256 vs
// global_head_dim 512), and gpu/residency.go's per-layer geometry seam (runLayer.ghd/gnKV) is
// never actually set anywhere in this backend — confirmed by grep, and by BuildResident failing
// on gemma4-dense-twogeom-tiny with "unsupported projection precision \"\"" even after this row's
// four features are declared. That is a SEPARATE, pre-existing gap (per-layer attention geometry
// on WebGPU) this row does not touch — CUDA/Metal both implement it, WebGPU does not yet.
func TestGemma3ResidentParityWebGPU(t *testing.T) {
	dir := "../testdata/gemma3-vl-tiny"
	// Stat the WEIGHTS, not the directory (audit-2026-09-10 G-13(h)). The dir and its config.json
	// are tracked while the weights are gitignored, so a dir stat passes on every clone. Then Load
	// failed on the missing weights, and the test skipped saying "no webgpu device".
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
	// A REAL numeric floor, not a smoke-test bar: 0.999, matched against a genuine checkpoint,
	// not a seeded/synthetic one — measured 0.9998 on this fixture at implementation time.
	if minCos < 0.999 {
		t.Errorf("minCosine %.6f < 0.999 — resident diverges from CPU on a real checkpoint", minCos)
	}
}

//go:build gpu && goinfer_testhooks

package gpu

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// gptOssParityPrompt is an arbitrary fixed token sequence, not tied to the fixture's tokenizer —
// same convention as gemma3ParityPrompt.
var gptOssParityPrompt = []int{1, 7, 42, 20, 5, 30, 13, 40}

// TestGptOssResidentParityWebGPU is G6's gpt-oss gate (docs/task-gpu-paths-2026-09.md):
// FeatAttnSink + FeatOutBias against the REAL, committed decoder/testdata/gptoss_tiny.gguf —
// not a seeded/synthetic fixture, so this is a genuine numeric floor, the same class of evidence
// gpu/gemma3_resident_parity_test.go established for the Gemma set.
//
// This is the hardest of G6's six features: it needed a per-head softmax sink threaded through
// every attention kernel (attn, attn-keys, attn-f16, attn-i8, and all three wide variants — 7
// pipelines), plus three brand-new MoE kernels (gpt-oss disagrees with the generic MoE path on
// what the router bias means and what the activation clamps — see routeGptOssWGSL/
// gptossGluQuantWGSL/moeExpertGptOssDownGEMVWGSL's own comments in gpu/moe.go).
//
// The floor was 0.95 (Metal's own tiny-fixture bar), and this backend measured 0.9943-0.9967
// against it. That margin was the defect. The fixture is bias-dominated (audit-2026-09-10 G-07),
// so C-06, where every int4 routed expert's down matmul collapsed to its bias, moved the logits only
// a few percent and passed. The bar is now cosine >= 0.998 with argmax matching at every position.
// It was registered before C-06's fix, from Metal's 0.9989 on the same fixture, and the 0.9943
// baseline fails it.
func TestGptOssResidentParityWebGPU(t *testing.T) {
	dir := "../decoder/testdata/gptoss_tiny.gguf"
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("no fixture (%s)", dir)
	}
	if !decoder.ResidentBackendFeatures("webgpu")[decoder.FeatAttnSink] {
		t.Skip("webgpu does not declare FeatAttnSink yet")
	}

	mg, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Skipf("no webgpu device: %v", err)
	}
	defer mg.Close()
	rf := mg.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("%s did not go resident (BuildResident refused) — decode path %q; decline: %s",
			dir, mg.DecodePath(), mg.ResidentDecline())
	}
	feats := mg.RequiredResidentFeatures()
	hasSink, hasOutBias := false, false
	for _, f := range feats {
		hasSink = hasSink || f == decoder.FeatAttnSink
		hasOutBias = hasOutBias || f == decoder.FeatOutBias
	}
	if !hasSink || !hasOutBias {
		t.Fatalf("fixture requires %v — this gate is only meaningful if it exercises both "+
			"FeatAttnSink and FeatOutBias; if the fixture changed, this test no longer gates "+
			"what it claims", feats)
	}
	t.Logf("resident decode path: %s", mg.DecodePath())

	mcpu, err := decoder.Load(dir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mcpu.Close()

	cache := mcpu.NewCache(len(gptOssParityPrompt))
	minCos, argmaxMiss := 1.0, 0
	argmaxLogits := func(v []float32) int {
		b := 0
		for i := range v {
			if v[i] > v[b] {
				b = i
			}
		}
		return b
	}
	for i, tok := range gptOssParityPrompt {
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
			if v != v || w != w {
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
		if a, b := argmaxLogits(cpuL), argmaxLogits(gpuL); a != b {
			argmaxMiss++
			t.Logf("  pos %2d argmax differs: cpu %d, resident %d", i, a, b)
		}
	}
	t.Logf("gpt-oss tiny, resident vs CPU (int4 both sides): minCosine=%.6f, argmax mismatches %d/%d",
		minCos, argmaxMiss, len(gptOssParityPrompt))
	if minCos < 0.998 || argmaxMiss > 0 {
		t.Errorf("minCosine %.6f (want >= 0.998), argmax mismatches %d (want 0) — the resident diverges "+
			"from CPU (bar registered before C-06's fix, from Metal's 0.9989 on this fixture; audit G-07)",
			minCos, argmaxMiss)
	}
}

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
// 0.95 is the SAME floor Metal's own gpt-oss tiny-fixture gate uses (metal/gptoss_real_test.go),
// not a looser bar invented for this backend — measured 0.9943-0.9967 across all 8 positions at
// implementation time, comfortably inside it.
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
	minCos := 1.0
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
	}
	t.Logf("gpt-oss tiny, resident vs CPU (int4 both sides): minCosine=%.6f", minCos)
	if minCos < 0.95 {
		t.Errorf("minCosine %.6f < 0.95 — resident diverges from CPU (same floor Metal's own "+
			"gpt-oss tiny-fixture gate uses)", minCos)
	}
}

//go:build gpu

package gpu_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/gpu"
)

// TestMLAResidency_realDeepSeek_R05 is the regression gate for review-finding R-05: a REAL
// DeepSeek MLA base (qk head dim = qk_nope 128 + qk_rope 64 = 192) must NOT be rejected by the
// resident decode path's 128-wide single-query-attention head-dim guard (attnHeadDimSupported).
// MLA runs its own mlaAttn kernels, so it is exempt from that guard; before the R-05 fix the guard
// declined every DeepSeek/Kimi base and residency silently fell back to CPU.
//
// The tiny fixture TestMLAResidency_matchesCPU uses has qkHead=48, so it does NOT exercise the >128
// path — this test needs a real DeepSeek. It is size-agnostic about the card: a 16B MoE won't fit an
// 8GB card fully resident (and MLA can't use the .giw streaming path), so residency may legitimately
// decline for VRAM. The gate is the DECLINE REASON, not fit: a head_dim decline means R-05 regressed;
// an OOM decline means the build got PAST the guard (the first check in newDecodeRunner) and R-05
// holds. Point GOINFER_MLA_MODEL at a DeepSeek/Kimi checkpoint (default: the box's V2-Lite GGUF).
func TestMLAResidency_realDeepSeek_R05(t *testing.T) {
	// The asset happening to be on disk is not a request to run a multi-GB test
	// (gpu/heavytest_test.go's requireHeavyModel, which this file — package gpu_test — cannot call
	// directly). Missing this gate let a checkpoint left over from earlier work make an ordinary
	// `-short` suite run try to load a real DeepSeek-V2-Lite and get killed (G-09's webgpu gate
	// group ran ./gpu/ automatically for the first time and caught it).
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
	}
	path := os.Getenv("GOINFER_MLA_MODEL")
	if path == "" {
		path = os.Getenv("HOME") + "/models/deepseek-v2-lite-gguf/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no DeepSeek/MLA checkpoint at %s (set GOINFER_MLA_MODEL): %v", path, err)
	}
	if _, err := gpu.New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}

	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int4"})
	if err != nil {
		t.Fatalf("Load(%s): %v", path, err)
	}
	defer m.Close()

	if m.ResidentActive() {
		// Resident is the R-05 gate; also decode a few tokens to confirm the resident MLA runner
		// actually RUNS (a model can report resident yet silently decode on CPU — the 7557723 seam).
		ch, _ := m.Generate(context.Background(), []int{1, 2, 3, 4, 5}, 8, decoder.SamplingParams{Temperature: 0})
		var toks []int
		for id := range ch {
			toks = append(toks, id)
		}
		if len(toks) == 0 {
			t.Fatalf("went resident but generated no tokens")
		}
		t.Logf("R-05 OK: real DeepSeek went webgpu-resident (MLA passed the head-dim admission and fit); "+
			"resident decode produced %d tokens %v", len(toks), toks)
		return
	}
	decline := m.ResidentDecline()
	t.Logf("not resident; decline reason: %q", decline)
	// R-05 regression: the 128-wide head-dim guard rejected MLA's 192-wide qk head.
	if strings.Contains(decline, "head_dim") {
		t.Fatalf("R-05 REGRESSED: MLA rejected by the head-dim guard (%q) — DeepSeek should be exempt", decline)
	}
	// Any other decline (VRAM OOM on this 8GB card, etc.) means the build got past the head-dim
	// guard — R-05 holds; the model just doesn't fit fully resident here (MLA can't stream).
	t.Logf("R-05 OK: MLA passed the head-dim guard; residency declined for a non-head-dim reason (%q)", decline)
}

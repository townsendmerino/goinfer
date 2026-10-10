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

// TestMLAResidency_realDeepSeek_R05 is the regression gate for a REAL DeepSeek MLA base (qk head
// dim = qk_nope 128 + qk_rope 64 = 192): it must NOT be rejected by attnHeadDimSupported, the
// resident decode path's single-query-attention head-dim guard. MLA runs its own mlaAttn kernels
// and is exempt from it; when the guard applied to MLA, every DeepSeek/Kimi base declined and
// residency silently fell back to CPU.
//
// The tiny fixture TestMLAResidency_matchesCPU uses (qkHead=48) does not exercise a real DeepSeek
// head size, so this test needs a real one. It is size-agnostic about the card: a 16B MoE won't
// fit an 8GB card fully resident (and MLA can't use the .giw streaming path), so residency may
// legitimately decline for VRAM. The gate is the DECLINE REASON, not fit: a head_dim decline
// means R-05 regressed, an OOM decline means the build got PAST the guard (the first check in
// newDecodeRunner). Point GOINFER_MLA_MODEL at a DeepSeek/Kimi checkpoint (default: the box's
// V2-Lite GGUF).
func TestMLAResidency_realDeepSeek_R05(t *testing.T) {
	// A model happening to be on disk is not a request to run a multi-GB test: the same opt-in as
	// requireHeavyModel (gpu/heavytest_test.go), which this file, package gpu_test, cannot call
	// directly. Without it a checkpoint left in ~/models made an ordinary `-short` run try to load a
	// real DeepSeek-V2-Lite.
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
		// actually RUNS (a model can report resident yet silently decode on CPU).
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
	// R-05 regression: the head-dim guard rejected MLA's 192-wide qk head.
	if strings.Contains(decline, "head_dim") {
		t.Fatalf("R-05 REGRESSED: MLA rejected by the head-dim guard (%q) — DeepSeek should be exempt", decline)
	}
	// Any other decline (VRAM OOM on this 8GB card, etc.) means the build got past the head-dim
	// guard, so it holds; the model just does not fit fully resident here (MLA can't stream).
	t.Logf("R-05 OK: MLA passed the head-dim guard; residency declined for a non-head-dim reason (%q)", decline)
}

package decoder

import (
	"context"
	"strings"
	"testing"
)

// TestEagleArchGuard_M07 gates M-07: the EAGLE entry points must refuse an own-runLayers
// family up front, not just recurrent/windowed ones. MLA (deepseek_v2/v3) is the canonical
// hole — it passes specRollbackSafe (its latent KV is truncatable) yet runLayersDeepseek never
// populates cache.captured, so the old code proceeded into the goroutine and fuseAt(0) sliced a
// nil buffer. canBatchN excludes MLA (the verify can't batch the latent-reconstruction layers),
// so the new canBatchN guard rejects it before the goroutine with a clear seam error. The caller
// then falls back to plain decode (lossless), rather than panicking mid-stream.
func TestEagleArchGuard_M07(t *testing.T) {
	// Minimal MLA model: reaches the canBatchN guard (greedy, rollback-safe, head hidden matches,
	// prompt long enough). No weights needed — the guard fires before any forward.
	m := &Model{w: &Weights{arch: &Architecture{Name: "deepseek_v2", HiddenDim: 8, NumLayers: 2, mla: &mlaParams{}}}}
	if !m.specRollbackSafe() {
		t.Fatal("precondition: MLA must be spec-rollback-safe (else an earlier guard, not M-07's, catches it)")
	}
	if m.canBatchN(6) {
		t.Fatal("precondition: MLA must be excluded from canBatchN")
	}
	head := &EagleHead{hidden: 8}
	prompt := []int{1, 2, 3}
	greedy := SamplingParams{Temperature: 0}
	ctx := context.Background()

	if _, _, err := m.GenerateEagleSpeculative(ctx, prompt, 4, head, []int{0}, 5, greedy); err == nil {
		t.Error("GenerateEagleSpeculative accepted an MLA target (M-07): should refuse, not panic in the goroutine")
	} else if !strings.Contains(err.Error(), "batched hidden-state capture not supported") {
		t.Errorf("GenerateEagleSpeculative: wrong refusal %q", err)
	}
	if _, _, err := m.GenerateEagleSpeculativeTree(ctx, prompt, 4, head, []int{0}, 2, 4, greedy); err == nil {
		t.Error("GenerateEagleSpeculativeTree accepted an MLA target (M-07)")
	} else if !strings.Contains(err.Error(), "batched hidden-state capture not supported") {
		t.Errorf("GenerateEagleSpeculativeTree: wrong refusal %q", err)
	}
}

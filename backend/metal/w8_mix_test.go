//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestW8Mix_M2_matchesCPU is int4mix's gate M2 (docs/tasks/task-metal-int8-2026-10.md, "Slice 4: int4mix"): in
// residentParity's harness (24 greedy steps in lockstep against the CPU at the same quant), Metal int4mix on its native
// path (attention W8A8, FFN int4) against the CPU's int4mix, beside the same with the native path off (the int4
// re-quant). Bar: the native arm's minimum cosine at least 0.99 and at least the re-quant arm's, and its hard flips (top-2
// gaps over 3%) no more than the re-quant arm's. GOINFER_W8_GATE_MODEL names another GGUF.
func TestW8Mix_M2_matchesCPU(t *testing.T) {
	path := w8GateModel(t)
	seed := seedPrompt(t, path, probeText)
	prev := nativeInt4Mix
	defer func() { nativeInt4Mix = prev }()
	nativeInt4Mix = true
	assertW8Attn(t, path, true)
	nat := residentParityAt(t, path, "int4mix", seed, 24)
	nativeInt4Mix = false
	assertW8Attn(t, path, false)
	req := residentParityAt(t, path, "int4mix", seed, 24)
	if nat.nan > 0 || req.nan > 0 {
		t.Fatalf("non-finite cosines: native %d, re-quant %d", nat.nan, req.nan)
	}
	t.Logf("M2: native min cosine %.6f, argmax %d/%d, hard flips %d; re-quant min cosine %.6f, argmax %d/%d, hard flips %d",
		nat.minCos, nat.exact, nat.steps, nat.hard, req.minCos, req.exact, req.steps, req.hard)
	if nat.minCos < 0.99 || nat.minCos < req.minCos {
		t.Errorf("M2 fails on cosine: native %.6f against the re-quant arm's %.6f and the 0.99 floor", nat.minCos, req.minCos)
	}
	if nat.hard > req.hard {
		t.Errorf("M2 fails on hard flips: native %d, re-quant %d", nat.hard, req.hard)
	}
}

// assertW8Attn loads path on Metal at int4mix and checks the resident took the native int4mix path exactly when want.
func assertW8Attn(t *testing.T, path string, want bool) {
	t.Helper()
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4mix"})
	if err != nil {
		t.Fatalf("load (metal, int4mix): %v", err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || a.r.w8Attn != want {
		t.Fatalf("int4mix: native path = %v, want %v (%s)", ok && a.r.w8Attn, want, m.DecodePath())
	}
}

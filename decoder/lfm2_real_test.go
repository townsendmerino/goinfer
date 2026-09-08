//go:build realckpt

// Real-model gate for LFM2.5-2.6B (model_type "lfm2", Lfm2ForCausalLM) — the T3 promotion of
// the lfm2 family from tiny-golden to a released checkpoint. LFM2.5-2.6B is the smallest
// released LFM2.5 checkpoint and the one an ad-hoc 2026-08-31 debugging session already diffed
// against by hand (decoder/lfm2_test.go's own comment: that pass found and fixed a wrong
// NormEps config key and a zeroed AttnScale) — this gate is the permanent, re-runnable version
// of that diff. Fixture: scripts/pin_lfm2_real.py.
//
//	go test -tags realckpt ./decoder/ -run TestLFM2Real -v -timeout 30m
package decoder

import (
	"encoding/json"
	"os"
	"testing"
)

func TestLFM2Real_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_LFM2_2_6B")
	const golden = "../testdata/lfm2_real_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_lfm2_real.py", err)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{}) // f32 resident → tight cosine
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()
	a := m.w.arch
	if a.Name != "lfm2" {
		t.Fatalf("arch = %q, want lfm2", a.Name)
	}
	// The real checkpoint's own layer_types is 22 conv / 8 full_attention (30 layers) — assert
	// the mix isn't degenerate, so a loader that silently classified every layer as one kind
	// would fail here rather than pass on a short prompt that never exercises the other kind.
	var conv, attn int
	for l := 0; l < a.NumLayers; l++ {
		if a.isConvLayer(l) {
			conv++
		} else {
			attn++
		}
	}
	t.Logf("lfm2.5-2.6b: %d layers (%d conv, %d attention), H=%d kv=%d headDim=%d", a.NumLayers, conv, attn, a.NumHeads, a.NumKVHeads, a.HeadDim)
	if conv == 0 || attn == 0 {
		t.Fatalf("layer-kind split degenerate: %d conv / %d attention — expected a mix", conv, attn)
	}

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("lfm2.5-2.6b parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 { // f32 vs f32 — tight
		t.Errorf("last-logit cosine %.6f < 0.9999", cos)
	}

	got := make([]int, 0, g.NNew)
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	t.Logf("lfm2.5-2.6b continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped on any
	// failure above). f32 vs f32 is the tightest oracle — argmax + continuation exact
	// when the gate passes, so argmax_pct is 100.
	emitParityRow(t, "lfm2", "full-forward-oracle", "HF f32 (LFM2.5-2.6B, Lfm2ForCausalLM)", 100.0, float64(cos), float64(cos))
}

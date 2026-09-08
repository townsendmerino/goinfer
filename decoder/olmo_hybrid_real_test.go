//go:build realckpt

// Real-model gate for Olmo Hybrid (model_type "olmo_hybrid", OlmoHybridForCausalLM) — the T3
// promotion of the olmo_hybrid family from tiny-golden to a released checkpoint.
// allenai/Olmo-Hybrid-7B is the only released size. Exercises per-layer NormPlacement (full-
// attention layers use olmo3's NormPostOnly, DeltaNet layers use plain NormPre2),
// whole-vector QK-norm on the full-attention layers, and linear_allow_neg_eigval on the
// DeltaNet layers. The release ships rope_parameters: {"rope_theta": null} — no RoPE anywhere,
// on any layer, so this family does not exercise the YaRN local/global split that olmo3's real
// gate found broken (see that gate's own comment); it is architecturally a different case, not
// a retest of the same one. Fixture: scripts/pin_olmo_hybrid_real.py.
//
//	go test -tags realckpt ./decoder/ -run TestOlmoHybridReal -v -timeout 30m
package decoder

import (
	"encoding/json"
	"os"
	"testing"
)

func TestOlmoHybridReal_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_OLMO_HYBRID_7B")
	const golden = "../testdata/olmo_hybrid_real_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_olmo_hybrid_real.py", err)
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
	if a.Name != "olmo_hybrid" {
		t.Fatalf("arch = %q, want olmo_hybrid", a.Name)
	}
	if a.qwen35 == nil {
		t.Fatalf("arch.qwen35 is nil, want the DeltaNet geometry set")
	}
	if !a.qwen35.NegEigval {
		t.Error("qwen35.NegEigval = false, want true (linear_allow_neg_eigval on the real release)")
	}
	if a.NormPlacement != NormPostOnly {
		t.Errorf("NormPlacement = %v, want NormPostOnly (full-attention layers)", a.NormPlacement)
	}
	if a.NormPlacementLinear == nil || *a.NormPlacementLinear != NormPre2 {
		t.Errorf("NormPlacementLinear = %v, want &NormPre2 (DeltaNet layers)", a.NormPlacementLinear)
	}
	if !a.QKNorm || !a.QKNormWhole {
		t.Errorf("QKNorm/QKNormWhole = %v/%v, want true/true", a.QKNorm, a.QKNormWhole)
	}
	var linear, full int
	for l := 0; l < a.NumLayers; l++ {
		if a.isLinearLayer(l) {
			linear++
		} else {
			full++
		}
	}
	t.Logf("olmo-hybrid-7b: %d layers (%d DeltaNet, %d full-attention), H=%d headDim=%d", a.NumLayers, linear, full, a.NumHeads, a.HeadDim)
	if linear == 0 || full == 0 {
		t.Fatalf("layer-kind split degenerate: %d linear / %d full — expected a 3:1 mix", linear, full)
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
	t.Logf("olmo-hybrid-7b parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
	t.Logf("olmo-hybrid-7b continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped on any
	// failure above). f32 vs f32 is the tightest oracle — argmax + continuation exact
	// when the gate passes, so argmax_pct is 100.
	emitParityRow(t, "olmo_hybrid", "full-forward-oracle", "HF f32 (allenai/Olmo-Hybrid-7B, OlmoHybridForCausalLM)", 100.0, float64(cos), float64(cos))
}

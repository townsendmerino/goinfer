//go:build realckpt

// Real-model gate for dense Granite 4.2 (model_type "granite", GraniteForCausalLM) — the T3
// promotion of the DENSE granite family from tiny-golden to a released checkpoint.
//
// NAME COLLISION WARNING: "granite" is NOT "granitemoehybrid" (the Mamba-2 + attention + MoE
// hybrid, already real-oracle via GOINFER_GRANITE_HF / TestGraniteReal_oracle in
// granite_real_test.go, which asserts arch.Name == "granitemoehybrid"). This gate targets the
// other, plain-llama-skeleton family and asserts arch.Name == "granite" explicitly so the two
// can never be confused by a future reader. Fixture: scripts/pin_granite_dense_real.py.
//
//	go test -tags realckpt ./decoder/ -run TestGraniteDenseReal -v -timeout 30m
package decoder

import (
	"encoding/json"
	"os"
	"testing"
)

func TestGraniteDenseReal_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_GRANITE_DENSE_3B")
	const golden = "../testdata/granite_dense_real_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_granite_dense_real.py", err)
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
	if a.Name != "granite" {
		t.Fatalf("arch = %q, want granite (NOT granitemoehybrid)", a.Name)
	}
	if a.MoE != nil {
		t.Fatalf("expected no MoE (dense Granite), got MoE config")
	}
	// The real release's one nontrivial scalar: attention_multiplier 0.015625 (embedding/
	// residual/logits multipliers are all 1.0 on every released 4.2 size).
	if got, want := a.AttnScale, 0.015625; got != want {
		t.Errorf("AttnScale = %v, want %v (attention_multiplier)", got, want)
	}
	t.Logf("granite-4.2-3b: %d layers, H=%d kv=%d headDim=%d attnScale=%v", a.NumLayers, a.NumHeads, a.NumKVHeads, a.HeadDim, a.AttnScale)

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("granite-4.2-3b parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
	t.Logf("granite-4.2-3b continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped on any
	// failure above). f32 vs f32 is the tightest oracle — argmax + continuation exact
	// when the gate passes, so argmax_pct is 100.
	emitParityRow(t, "granite", "full-forward-oracle", "HF f32 (ibm-granite/granite-4.2-3b, GraniteForCausalLM)", 100.0, float64(cos), float64(cos))
}

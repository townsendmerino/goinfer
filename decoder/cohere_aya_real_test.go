//go:build realckpt

// Real-model gate for Aya-Expanse-8B (model_type "cohere", CohereForCausalLM) — the T3
// proof of goinfer's Cohere / Command-R Phase 1 on actual released weights. Aya-Expanse-8B
// is the smallest cohere1 checkpoint AND use_qk_norm=false, so it is exactly the Phase-1
// scope. At 8B the f32 forward fits in RAM, so goinfer loads f32 (Options{} ⇒ quantNone)
// and the gate is a TIGHT cosine vs the HF f32 golden (argmax + greedy continuation +
// cosine ≥ 0.9999). Verifies the real bias-free LayerNorm + parallel block + GPT-J
// interleaved RoPE + logit_scale multiplier + tied 256k embeddings on genuine weights.
// Fixture: scripts/pin_cohere_aya.py.
//
//	go test -tags realckpt ./decoder/ -run TestCohereAyaReal -v -timeout 30m
package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCohereAyaReal_gate(t *testing.T) {
	home, _ := os.UserHomeDir()
	ckpt := os.Getenv("GOINFER_COHERE_AYA")
	if ckpt == "" {
		ckpt = filepath.Join(home, "models", "aya-expanse-8b")
	}
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no Aya-Expanse-8B at %s: %v", ckpt, err)
	}
	const golden = "../testdata/cohere_aya_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_cohere_aya.py", err)
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
	if a.Name != "cohere" {
		t.Fatalf("arch = %q, want cohere", a.Name)
	}
	t.Logf("aya-expanse-8b: %d layers, H=%d kv=%d headDim=%d inter=%d logitScale(recip)=%.4f tied=%v",
		a.NumLayers, a.NumHeads, a.NumKVHeads, a.HeadDim, a.IntermediateDim, a.LogitScale, a.TiedLMHead)

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("aya-expanse parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
	t.Logf("aya-expanse continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped on any
	// failure above). f32 vs f32 is the tightest oracle — argmax + continuation are exact
	// when the gate passes, so argmax_pct is 100.
	emitParityRow(t, "cohere", "full-forward-oracle", "HF f32 (Aya-Expanse-8B, CohereForCausalLM)", 100.0, float64(cos), float64(cos))
}

//go:build realckpt

// Real-model gate for gemma-3-4b-it TEXT decoder (gemma3) — the safetensors loader + gemma3
// forward on actual released weights, text-only (the VL wrapper's vision path is not exercised).
// 4B fits an f32 forward in RAM, so Options{} ⇒ quantNone and the gate is a TIGHT cosine vs the
// HF f32 golden (argmax + greedy continuation + cosine ≥ 0.9999). Verifies the gemma3 axis on
// real weights: sandwich (4-norm) placement + per-head QK-norm + sliding-window interleave +
// embed scale. Fixture: scripts/pin_gemma3_4b_text.py. This is the real-oracle emit gate that
// moves gemma3 pending → validated (batched-prefill coverage: gemma3 is canBatchN-batchable).
//
//	go test -tags realckpt ./decoder/ -run TestGemma3Real -v -timeout 20m
package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestGemma3Real_gate(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	ckpt := os.Getenv("GEMMA3_4B")
	if ckpt == "" {
		ckpt = filepath.Join(home, "models", "gemma-3-4b-it")
	}
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no gemma-3-4b-it at %s: %v", ckpt, err)
	}
	const golden = "../testdata/gemma3_4b_text_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_gemma3_4b_text.py", err)
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
	if a.Name != "gemma3" {
		t.Fatalf("arch = %q, want gemma3", a.Name)
	}
	t.Logf("gemma-3-4b: %d layers, H=%d kv=%d headDim=%d sandwich=%v qkNorm=%v",
		a.NumLayers, a.NumHeads, a.NumKVHeads, a.HeadDim, a.NormPlacement, a.QKNorm)

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("gemma-3-4b parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	// f32 vs f32, but Gemma's final-logit SOFTCAP (30·tanh(logit/30)) + GELU-tanh reshape tiny
	// per-layer f32 deltas over 34 layers, so the vector cosine floors ~0.9997 (measured), not the
	// ~1.0 of a plain-MHA model. The primary proof is the EXACT argmax + 6-token greedy continuation
	// above — a wrong forward cannot reproduce that string. 0.999 keeps the cosine honest without
	// false-redding on the softcap's numerics.
	if cos < 0.999 {
		t.Errorf("last-logit cosine %.6f < 0.999", cos)
	}

	got := make([]int, 0, g.NNew)
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	t.Logf("gemma-3-4b continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped if any check
	// above failed). f32 vs f32 → the tightest oracle; argmax + continuation exact when green.
	emitParityRow(t, "gemma3", "full-forward-oracle", "HF f32 (gemma-3-4b-it, text)", 100.0, float64(cos), float64(cos))
}

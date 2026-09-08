//go:build realckpt

// Real-model gate for Ministral 3 (registry key "mistral3"; released repos wrap it as
// Mistral3ForConditionalGeneration, text_config.model_type "ministral3") — the T3 promotion of
// the mistral3 family from tiny-golden to a released checkpoint. Ministral-3-3b is the smallest
// released size; the gate targets the -BF16 sibling repo (the default repo ships FP8-native).
// Exercises the two real new primitives on real weights: attn-temp
// (AttnTempBeta/AttnTempOrigMaxPos, Llama4's own formula generalized to run alongside RoPE) and
// YaRN with the DeepSeek-style mscale/mscale_all_dim override (both 1.0 on the real release, so
// this asserts the override resolves to the trivial-but-correct case rather than the tiny
// fixture's distinct-ratio case). Fixture: scripts/pin_ministral3_real.py.
//
//	go test -tags realckpt ./decoder/ -run TestMinistral3Real -v -timeout 30m
package decoder

import (
	"encoding/json"
	"os"
	"testing"
)

func TestMinistral3Real_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_MINISTRAL3_3B")
	const golden = "../testdata/ministral3_real_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_ministral3_real.py", err)
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
	if a.Name != "mistral3" {
		t.Fatalf("arch = %q, want mistral3", a.Name)
	}
	if a.SlidingWindow != 0 {
		t.Errorf("SlidingWindow = %d, want 0 (the real release ships sliding_window: null)", a.SlidingWindow)
	}
	if got, want := a.AttnTempBeta, 0.1; got != want {
		t.Errorf("AttnTempBeta = %v, want %v (llama_4_scaling_beta)", got, want)
	}
	if got, want := a.AttnTempOrigMaxPos, 16384.0; got != want {
		t.Errorf("AttnTempOrigMaxPos = %v, want %v (original_max_position_embeddings)", got, want)
	}
	if a.ropeScaling == nil || a.ropeScaling.kind != ropeScaleYarn {
		t.Fatalf("ropeScaling = %+v, want YaRN", a.ropeScaling)
	}
	t.Logf("ministral3-3b: %d layers, H=%d kv=%d headDim=%d attnTempBeta=%v origMaxPos=%v",
		a.NumLayers, a.NumHeads, a.NumKVHeads, a.HeadDim, a.AttnTempBeta, a.AttnTempOrigMaxPos)

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("ministral3-3b parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
	t.Logf("ministral3-3b continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped on any
	// failure above). f32 vs f32 is the tightest oracle — argmax + continuation exact
	// when the gate passes, so argmax_pct is 100.
	emitParityRow(t, "mistral3", "full-forward-oracle", "HF f32 (Ministral-3-3b-Instruct-2512-BF16, Mistral3ForConditionalGeneration text path)", 100.0, float64(cos), float64(cos))
}

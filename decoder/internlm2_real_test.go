//go:build realckpt

// Real-model gate for InternLM2.5-1.8B-Chat (model_type "internlm2",
// InternLM2ForCausalLM) — the T3 promotion of the internlm2 family from tiny-golden to a
// released checkpoint. The tiny fixture (internlm2_test.go) pins the grouped wqkv
// de-interleave in isolation, against a llama repacked BY HAND into InternLM2's fused
// layout; this gate proves the same de-interleave against the model's own NATIVE fused
// wqkv weights, on real trained values, end to end. Fixture: scripts/pin_internlm2_real.py.
//
//	go test -tags realckpt ./decoder/ -run TestInternLM2_1_8bReal_gate -v -timeout 30m
package decoder

import (
	"encoding/json"
	"os"
	"testing"
)

func TestInternLM2_1_8bReal_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_INTERNLM2_1_8B")
	const golden = "../testdata/internlm2_real_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_internlm2_real.py", err)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
		Groups          int       `json:"groups"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if g.Groups < 2 {
		t.Fatalf("fixture has groups=%d: with groups<2 the grouped layout and a plain concat "+
			"coincide, so this gate would pass against the wrong reader", g.Groups)
	}

	m, err := Load(ckpt, Options{}) // f32 resident → tight cosine
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()
	a := m.w.arch
	if a.Name != "internlm2" {
		t.Fatalf("arch = %q, want internlm2", a.Name)
	}
	// The split must produce the ordinary per-projection shapes the generic forward expects
	// (same check as the tiny fixture's gate, run here against the real checkpoint's geometry).
	if got := m.w.Layers[0].QProj.Rows(); got != a.NumHeads*a.HeadDim {
		t.Errorf("q rows = %d, want %d", got, a.NumHeads*a.HeadDim)
	}
	if got := m.w.Layers[0].KProj.Rows(); got != a.NumKVHeads*a.HeadDim {
		t.Errorf("k rows = %d, want %d", got, a.NumKVHeads*a.HeadDim)
	}
	t.Logf("internlm2-1.8b: %d layers, H=%d kv=%d headDim=%d groups=%d",
		a.NumLayers, a.NumHeads, a.NumKVHeads, a.HeadDim, g.Groups)

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("internlm2-1.8b parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
	t.Logf("internlm2-1.8b continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped on any
	// failure above). f32 vs f32 is the tightest oracle — argmax + continuation exact
	// when the gate passes, so argmax_pct is 100.
	emitParityRow(t, "internlm2", "full-forward-oracle", "HF f32 (internlm2_5-1_8b-chat, InternLM2ForCausalLM)", 100.0, float64(cos), float64(cos))
}

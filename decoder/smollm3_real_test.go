//go:build realckpt

// Real-model gate for SmolLM3-3B (model_type "smollm3", SmolLM3ForCausalLM) — the T3
// promotion of the smollm3 family from tiny-golden to a released checkpoint. SmolLM3-3B
// is the smallest AND only released size in the family, so it fits in RAM at f32 and the
// gate is a TIGHT cosine vs the HF f32 golden (argmax + greedy continuation + cosine >=
// 0.9999). Fixture: scripts/pin_smollm3_3b.py.
//
//	go test -tags realckpt ./decoder/ -run TestSmolLM3_3bReal -v -timeout 30m
package decoder

import (
	"encoding/json"
	"os"
	"testing"
)

func TestSmolLM3_3bReal_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_SMOLLM3_3B")
	const golden = "../testdata/smollm3_3b_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_smollm3_3b.py", err)
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
	if a.Name != "smollm3" {
		t.Fatalf("arch = %q, want smollm3", a.Name)
	}
	// The real checkpoint's own no_rope_layers is [1,1,1,0] repeating (36 layers, interval 4):
	// every 4th layer (index 3, 7, 11, ...) is NoPE, the rest carry RoPE. Same convention the
	// tiny fixture pins (smollm3_test.go) — assert it here too so a released-checkpoint config
	// surprise (a different interval, or the field missing) is visible as a test failure, not
	// silently absorbed into a degenerate all-RoPE forward.
	var nope int
	for l := 0; l < a.NumLayers; l++ {
		want := (l+1)%4 == 0
		got := a.isNoPELayer(l)
		if got != want {
			t.Errorf("layer %d: isNoPELayer=%v, want %v (no_rope_layers repeats [1,1,1,0])", l, got, want)
		}
		if got {
			nope++
		}
	}
	t.Logf("smollm3-3b: %d layers (%d NoPE), H=%d kv=%d headDim=%d", a.NumLayers, nope, a.NumHeads, a.NumKVHeads, a.HeadDim)
	if nope == 0 || nope == a.NumLayers {
		t.Fatalf("NoPE pattern degenerate: %d/%d NoPE layers — expected a mix", nope, a.NumLayers)
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
	t.Logf("smollm3-3b parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
	t.Logf("smollm3-3b continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped on any
	// failure above). f32 vs f32 is the tightest oracle — argmax + continuation exact
	// when the gate passes, so argmax_pct is 100.
	emitParityRow(t, "smollm3", "full-forward-oracle", "HF f32 (SmolLM3-3B, SmolLM3ForCausalLM)", 100.0, float64(cos), float64(cos))
}

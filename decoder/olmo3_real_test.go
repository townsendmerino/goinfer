//go:build realckpt

// Real-model gate for Olmo 3 (model_type "olmo3", Olmo3ForCausalLM) — the T3 promotion of the
// olmo3 family from tiny-golden to a released checkpoint. allenai/Olmo-3-7B-Think is the
// smallest released size. Exercises Olmo 3's two real departures on real weights: NormPostOnly
// (no pre-norm at all) and QKNormWhole (QK-norm over the full projected q/k vector, not
// per-head), plus the local/global RoPE split (sliding/full 3:1, YaRN on full_attention layers
// only). Fixture: scripts/pin_olmo3_real.py.
//
//	go test -tags realckpt ./decoder/ -run TestOlmo3Real -v -timeout 30m
package decoder

import (
	"encoding/json"
	"os"
	"testing"
)

func TestOlmo3Real_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_OLMO3_7B")
	const golden = "../testdata/olmo3_real_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_olmo3_real.py", err)
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
	if a.Name != "olmo3" {
		t.Fatalf("arch = %q, want olmo3", a.Name)
	}
	if a.NormPlacement != NormPostOnly {
		t.Errorf("NormPlacement = %v, want NormPostOnly", a.NormPlacement)
	}
	if !a.QKNorm || !a.QKNormWhole {
		t.Errorf("QKNorm/QKNormWhole = %v/%v, want true/true", a.QKNorm, a.QKNormWhole)
	}
	wantQNormLen := a.NumHeads * a.HeadDim
	if got := len(m.w.Layers[0].QNorm); got != wantQNormLen {
		t.Errorf("layer 0 QNorm length = %d, want %d (num_heads*head_dim)", got, wantQNormLen)
	}
	// The real release is sliding/full 3:1 (sliding_window=4096) — assert the mix isn't
	// degenerate, same discipline as cohere2's interleave assertion.
	var global int
	for l := 0; l < a.NumLayers; l++ {
		if a.isGlobalLayer(l) {
			global++
		}
	}
	t.Logf("olmo3-7b: %d layers (%d full/global, %d sliding), H=%d kv=%d headDim=%d window=%d",
		a.NumLayers, global, a.NumLayers-global, a.NumHeads, a.NumKVHeads, a.HeadDim, a.SlidingWindow)
	if global == 0 || global == a.NumLayers {
		t.Fatalf("interleave degenerate: %d/%d global layers — expected a 3:1 mix", global, a.NumLayers)
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
	t.Logf("olmo3-7b parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
	t.Logf("olmo3-7b continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped on any
	// failure above). f32 vs f32 is the tightest oracle — argmax + continuation exact
	// when the gate passes, so argmax_pct is 100.
	emitParityRow(t, "olmo3", "full-forward-oracle", "HF f32 (allenai/Olmo-3-7B-Think, Olmo3ForCausalLM)", 100.0, float64(cos), float64(cos))
}

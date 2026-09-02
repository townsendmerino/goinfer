//go:build realckpt

// Real-model gate for Command-R7B (model_type "cohere2", Cohere2ForCausalLM) — the T3
// proof of goinfer's Cohere2 / Command-R7B on actual released weights. Command-R7B is the
// smallest cohere2 checkpoint; it exercises the Phase-2 primitives on real weights:
// interleaved sliding-window / full attention (every 4th layer global) and NoPE on the
// global layers (only the sliding layers rope). At 7B the f32 forward fits in RAM, so
// goinfer loads f32 (Options{} ⇒ quantNone) and the gate is a TIGHT cosine vs the HF f32
// golden (argmax + greedy continuation + cosine ≥ 0.9999). Fixture: scripts/pin_cohere2_r7b.py.
//
//	go test -tags realckpt ./decoder/ -run TestCohere2R7bReal -v -timeout 30m
package decoder

import (
	"encoding/json"
	"os"
	"testing"
)

func TestCohere2R7bReal_gate(t *testing.T) {
	requireHeavyModel(t)
	// assetPath, not a hand-rolled env+fallback. os.Stat on the DIRECTORY was the weaker check the
	// registry exists to replace: a directory that exists but holds no shards satisfied it.
	ckpt := assetPath(t, "GOINFER_COHERE2_R7B")
	const golden = "../testdata/cohere2_r7b_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_cohere2_r7b.py", err)
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
	if a.Name != "cohere2" {
		t.Fatalf("arch = %q, want cohere2", a.Name)
	}
	// The Phase-2 interleave on the REAL model: some layers must be global (full) and
	// NoPE while the rest are sliding + RoPE. A degenerate all-global / all-rope
	// classification would silently pass the cosine on short prompts, so assert it holds.
	var global, nope int
	for l := 0; l < a.NumLayers; l++ {
		if a.isGlobalLayer(l) {
			global++
		}
		if a.isNoPELayer(l) {
			nope++
		}
		if a.isGlobalLayer(l) != a.isNoPELayer(l) {
			t.Errorf("layer %d: isGlobal=%v but isNoPE=%v (cohere2: global ⇔ NoPE)", l, a.isGlobalLayer(l), a.isNoPELayer(l))
		}
	}
	t.Logf("command-r7b: %d layers (%d global/NoPE, %d sliding/RoPE), H=%d kv=%d headDim=%d window=%d logitScale(recip)=%.4f",
		a.NumLayers, global, nope, a.NumHeads, a.NumKVHeads, a.HeadDim, a.SlidingWindow, a.LogitScale)
	if global == 0 || global == a.NumLayers {
		t.Fatalf("interleave degenerate: %d/%d global layers — expected a mix (sliding + every-Nth global)", global, a.NumLayers)
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
	t.Logf("command-r7b parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
	t.Logf("command-r7b continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped on any
	// failure above). f32 vs f32 is the tightest oracle — argmax + continuation exact
	// when the gate passes, so argmax_pct is 100.
	emitParityRow(t, "cohere2", "full-forward-oracle", "HF f32 (Command-R7B, Cohere2ForCausalLM)", 100.0, float64(cos), float64(cos))
}

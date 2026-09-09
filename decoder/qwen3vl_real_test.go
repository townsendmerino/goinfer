//go:build realckpt

// Real-model gate for Qwen3-VL-2B-Instruct (qwen3_vl, TEXT-ONLY — P8 Phase 0, docs/multimodal.md):
// the fused Qwen3-shaped attention (per-head q/k RMSNorm, GQA, no q/k/v bias) wired under
// Qwen3-VL's nested text_config, on actual released weights. No pixel_values, no vision tower, no
// DeepStack — those are explicitly out of scope for this phase. At 2B the f32 forward fits in RAM,
// so the gate is a TIGHT cosine vs the HF f32 golden (mirrors decoder/phi3_real_test.go's own
// reasoning). Fixture: scripts/pin_qwen3vl_real.py; asset: GOINFER_QWEN3VL_2B
// (testdata/assets.json) — not present on this box as of 2026-09-08, so this gate SKIPS cleanly
// rather than being silently absent from the registry.
//
//	go test -tags realckpt ./decoder/ -run TestQwen3VLReal_gate -v -timeout 20m
package decoder

import (
	"encoding/json"
	"os"
	"testing"
)

func TestQwen3VLReal_gate(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_QWEN3VL_2B")
	const golden = "../testdata/qwen3vl_real_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_qwen3vl_real.py", err)
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
	if a.Name != "qwen3_vl" {
		t.Fatalf("arch = %q, want qwen3_vl", a.Name)
	}
	if !a.MRopeInterleaved {
		t.Error("arch.MRopeInterleaved = false, want true for qwen3_vl")
	}
	t.Logf("qwen3-vl-2b: %d layers, H=%d kv=%d headDim=%d inter=%d MRopeSection=%v",
		a.NumLayers, a.NumHeads, a.NumKVHeads, a.HeadDim, a.IntermediateDim, a.MRopeSection)

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("qwen3-vl-2b text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
	t.Logf("qwen3-vl-2b continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	emitParityRow(t, "qwen3_vl", "full-forward-oracle", "HF f32 (Qwen3-VL-2B-Instruct, text-only prefill)", 100.0, float64(cos), float64(cos))
}

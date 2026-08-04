//go:build realckpt

// Real-model gate for TinyLlama-1.1B-Chat (llama, dense) — the safetensors loader + llama forward on
// actual released weights. 1.7B fits an f32 forward in RAM, so Options{} ⇒ quantNone and the
// gate is a TIGHT cosine vs the HF f32 golden (argmax + greedy continuation + cosine ≥ 0.9999),
// not the int8-vs-bf16 of the larger families. Verifies the llama axis on real weights:
// GQA + QKV-bias + full rotary. Fixture: scripts/pin_llama_real.py.
// This is the real-oracle emit gate that moves llama pending → validated in the parity manifest
// (batched-prefill coverage: llama is canBatchN-batchable).
//
//	go test -tags realckpt ./decoder/ -run TestLlamaReal -v -timeout 20m
package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestLlamaReal_gate(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	ckpt := os.Getenv("GOINFER_QWEN3_REAL")
	if ckpt == "" {
		ckpt = filepath.Join(home, "models", "tinyllama-1.1b-chat")
	}
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no TinyLlama-1.1B-Chat at %s: %v", ckpt, err)
	}
	const golden = "../testdata/llama_real_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_llama_real.py", err)
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
	if a.Name != "llama" {
		t.Fatalf("arch = %q, want llama", a.Name)
	}
	t.Logf("TinyLlama-1.1B: %d layers, H=%d kv=%d headDim=%d inter=%d qkvBias tied=%v",
		a.NumLayers, a.NumHeads, a.NumKVHeads, a.HeadDim, a.IntermediateDim, a.TiedLMHead)

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("TinyLlama-1.1B parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
	t.Logf("TinyLlama-1.1B continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
	// Record the validated metrics (no-op unless GOINFER_MANIFEST_EMIT; skipped if any check
	// above failed). f32 vs f32 → the tightest oracle; argmax + continuation exact when green.
	emitParityRow(t, "llama", "full-forward-oracle", "HF f32 (TinyLlama-1.1B-Chat)", 100.0, float64(cos), float64(cos))
}

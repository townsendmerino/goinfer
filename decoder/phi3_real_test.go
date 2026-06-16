//go:build realckpt

// Real-model gate for Phi-3-mini-4k-instruct (phi3, 3.8B dense) — the fused-tensor loader
// + generic llama forward on actual released weights. At 3.8B the f32 forward fits in RAM,
// so goinfer loads f32 (Options{} ⇒ quantNone) and the gate is a TIGHT cosine vs the HF
// f32 golden (argmax + greedy continuation + cosine ≥ 0.9999) — not the int8-vs-bf16 of the
// larger families. Verifies the qkv_proj / gate_up_proj split + full rotary on real weights.
// Fixture: scripts/pin_phi3_mini.py.
//
//	go test -tags realckpt ./decoder/ -run TestPhi3MiniReal -v -timeout 20m
package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestPhi3MiniReal_gate(t *testing.T) {
	home, _ := os.UserHomeDir()
	ckpt := os.Getenv("GOINFER_PHI3_MINI")
	if ckpt == "" {
		ckpt = filepath.Join(home, "models", "phi3-mini-4k")
	}
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no Phi-3-mini-4k at %s: %v", ckpt, err)
	}
	const golden = "../testdata/phi3_mini_golden.json"
	raw, err := os.ReadFile(golden)
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_phi3_mini.py", err)
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
	if a.Name != "phi3" {
		t.Fatalf("arch = %q, want phi3", a.Name)
	}
	t.Logf("phi3-mini: %d layers, H=%d kv=%d headDim=%d inter=%d rotaryDim=%d tied=%v",
		a.NumLayers, a.NumHeads, a.NumKVHeads, a.HeadDim, a.IntermediateDim, a.RotaryDim, a.TiedLMHead)

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("phi3-mini parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
	t.Logf("phi3-mini continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, got[i], g.ContinuationIDs[i])
			break
		}
	}
}

// TestPhi3GGUFReal_gate exercises the llama.cpp phi3 GGUF loader on Phi-3-mini-4k q4: the
// fused attn_qkv → q/k/v and ffn_up → gate/up splits, the omitted-rope.freq_base default,
// and partial-rotary handling. It matches the last-token argmax + greedy continuation
// against the f32 golden — the real proof the loader is correct: a wrong fused split could
// not reproduce a string of exact argmaxes. Cosine is LOGGED, not asserted: the microsoft
// GGUF is an earlier Phi-3 snapshot than the (updated Phi-3.1) safetensors golden — same
// architecture, slightly different weights — so the full-vector cosine (~0.75) reflects the
// snapshot delta + Q4, not load error (f32-resident vs int8-resident agree to ~0.02, ruling
// out goinfer quant). Fixture reuses scripts/pin_phi3_mini.py's golden.
func TestPhi3GGUFReal_gate(t *testing.T) {
	home, _ := os.UserHomeDir()
	gguf := os.Getenv("GOINFER_PHI3_GGUF")
	if gguf == "" {
		gguf = filepath.Join(home, "models", "phi3-mini-4k-gguf", "Phi-3-mini-4k-instruct-q4.gguf")
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no Phi-3 GGUF at %s: %v", gguf, err)
	}
	raw, err := os.ReadFile("../testdata/phi3_mini_golden.json")
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_phi3_mini.py", err)
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
	m, err := Load(gguf, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load(%s): %v", gguf, err)
	}
	defer m.Close()
	if m.w.arch.Name != "phi3" {
		t.Fatalf("arch = %q, want phi3", m.w.arch.Name)
	}
	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("phi3 GGUF parity: argmax got=%d want=%d | cosine=%.4f (logged; Q4 + earlier snapshot)", argmax(logits), g.Argmax, cos)
	if argmax(logits) != g.Argmax {
		t.Errorf("GGUF argmax = %d, want %d", argmax(logits), g.Argmax)
	}
	got := make([]int, 0, g.NNew)
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	t.Logf("phi3 GGUF continuation got=%v want=%v", got, g.ContinuationIDs)
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("GGUF continuation[%d] = %d, want %d (got %v)", i, got[i], g.ContinuationIDs[i], got)
			break
		}
	}
}

package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestGemma4MoE_forwardParity is the Phase-2b end-to-end gate: goinfer's whole
// gemma4 forward — Load of the tiny enable_moe_block checkpoint (the safetensors
// loader's gemma4 branch: dense MLP + layer_scalar + the parallel MoE sub-block)
// driving runLayersGemma4 with gemma4MoEFFN wired in — against the HF oracle in
// scripts/pin_gemma4_moe_forward.py. f32 weights, so this is a TIGHT gate: the
// last-token argmax must match, the full-vocab logits must match by cosine, and
// the greedy continuation must be byte-identical. The scaling params (norms /
// router.scale / per_expert_scale / layer_scalar) are strengthened in the
// generator, so an identity/no-op bug in how goinfer applies them can't hide.
func TestGemma4MoE_forwardParity(t *testing.T) {
	raw, err := os.ReadFile("../testdata/gemma4_moe_forward_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_gemma4_moe_forward.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float64 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	const ckpt = "../testdata/gemma4-moe-tiny"
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_gemma4_moe_forward.py", ckpt)
	}
	m, err := Load(ckpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	// Last-token logits over the full prompt.
	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	for _, id := range g.PromptIDs[:len(g.PromptIDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("prefill runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.PromptIDs[len(g.PromptIDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(logits) != len(g.LastLogits) {
		t.Fatalf("got %d logits, want %d", len(logits), len(g.LastLogits))
	}

	got := argmax(logits)
	t.Logf("argmax: got %d (logit %.4f) | want %d (golden logit %.4f)",
		got, logits[got], g.Argmax, logits[g.Argmax])
	if got != g.Argmax {
		t.Errorf("argmax = %d, want %d", got, g.Argmax)
	}

	// Full-vocab cosine + maxAbs (f32 checkpoint vs fp32 oracle — tight).
	var dot, na, nb, maxAbs float64
	for i, want := range g.LastLogits {
		a := float64(logits[i])
		if d := math.Abs(a - want); d > maxAbs {
			maxAbs = d
		}
		dot += a * want
		na += a * a
		nb += want * want
	}
	cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
	t.Logf("last-logit parity: cosine=%.8f maxAbs=%.3e over %d vocab", cos, maxAbs, len(g.LastLogits))
	if cos < 0.9999 {
		t.Errorf("cosine %.8f < 0.9999", cos)
	}
	if maxAbs > 5e-2 {
		t.Errorf("maxAbs %.3e > 5e-2", maxAbs)
	}

	// Greedy continuation must be byte-identical (argmax at every step).
	cur := append([]int(nil), g.PromptIDs...)
	cont := make([]int, 0, g.NNew)
	for range g.NNew {
		c2 := m.NewCache(len(cur))
		for _, id := range cur[:len(cur)-1] {
			if _, err := m.runLayers(id, c2); err != nil {
				t.Fatalf("continuation runLayers: %v", err)
			}
		}
		lg, err := m.forward(cur[len(cur)-1], c2)
		if err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
		nxt := argmax(lg)
		cont = append(cont, nxt)
		cur = append(cur, nxt)
	}
	t.Logf("continuation: got %v | want %v", cont, g.ContinuationIDs)
	for i := range cont {
		if i >= len(g.ContinuationIDs) || cont[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d", i, cont[i], g.ContinuationIDs[i])
			break
		}
	}
}

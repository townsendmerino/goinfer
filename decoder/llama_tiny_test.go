package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestLlamaTiny_textParity pins the plain `llama` forward against a tiny-random HF oracle.
// The baseline architecture needs its own tiny fixture: `llama` is a required parity gate and the most common
// architecture, and the only other llama checkpoints are gitignored on the Linux box, so no other machine could
// exercise it and the .giw census round-tripped other families without ever touching it (audit-2026-09-02 C-03).
//
// Deliberately plain: GQA 2:1, SwiGLU, RMSNorm, UNTIED head, rope_theta at Llama-3's 500000.0. Every other family's
// descriptor is a deviation from this one, so a break here is a break in the thing they all deviate FROM. The untied
// head matters on its own: it is a separate LMHead tensor the serializer carries, which tied families cannot exercise.
func TestLlamaTiny_textParity(t *testing.T) {
	raw, err := os.ReadFile("../testdata/llama_tiny_text_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_llama_tiny.py")
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
	const ckpt = "../testdata/llama-tiny"
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_llama_tiny.py", ckpt)
	}
	m, err := Load(ckpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if m.w.arch.Name != "llama" {
		t.Fatalf("arch = %q, want llama — the fixture is meant to pin the BASELINE arch", m.w.arch.Name)
	}
	if m.w.arch.TiedLMHead {
		t.Fatal("fixture resolved as tied; it is built untied on purpose, so a tied read here " +
			"means the config was misparsed and the LMHead path is not being exercised")
	}

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
	if got != g.Argmax {
		t.Errorf("argmax = %d, want %d", got, g.Argmax)
	}
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
	t.Logf("llama-tiny parity: argmax=%d cosine=%.8f maxAbs=%.3e over %d vocab",
		got, cos, maxAbs, len(g.LastLogits))
	if cos < 0.9999 {
		t.Errorf("cosine %.8f < 0.9999", cos)
	}
	if maxAbs > 5e-2 {
		t.Errorf("maxAbs %.3e > 5e-2", maxAbs)
	}

	// A MATCHING ARGMAX MEANS LITTLE ON ITS OWN — both LFM2 bugs held argmax while the logit
	// cosine was 0.897. The greedy continuation is the part that compounds.
	cur := append([]int(nil), g.PromptIDs...)
	cont := make([]int, 0, g.NNew)
	c2 := m.NewCache(len(cur) + g.NNew)
	var lg []float32
	for _, id := range cur {
		if lg, err = m.forward(id, c2); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	for range g.NNew {
		id := argmax(lg)
		cont = append(cont, id)
		if lg, err = m.forward(id, c2); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	for i := range g.ContinuationIDs {
		if i >= len(cont) || cont[i] != g.ContinuationIDs[i] {
			t.Fatalf("greedy continuation diverges at %d: got %v, want %v", i, cont, g.ContinuationIDs)
		}
	}
}

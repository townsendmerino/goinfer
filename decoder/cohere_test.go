package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestCohere_forwardParity is the Phase-1 end-to-end gate for the Cohere /
// Command-R family (docs/task-model-family-cohere.md). It drives goinfer's whole
// cohere forward — Load of the tiny safetensors checkpoint through the generic
// loader (cohereTensorSchema) into the NormParallel forward path — against the HF
// CohereForCausalLM oracle in scripts/pin_cohere_tiny.py. f32 weights, so this is
// a TIGHT gate: last-token argmax exact, full-vocab logits by cosine, and the
// greedy continuation byte-identical.
//
// It pins the two new primitives (bias-free LayerNorm; the parallel attn+MLP
// block) plus the borrowed ones (GPT-J interleaved RoPE; the logit_scale
// multiply, stored as goinfer's reciprocal). The generator reseeds every
// LayerNorm weight to non-trivial values, so a bug that drops the norm weight
// (×1) can't hide behind HF's identity init.
//
// Regenerate:  python3 scripts/pin_cohere_tiny.py
func TestCohere_forwardParity(t *testing.T) {
	raw, err := os.ReadFile("../testdata/cohere_tiny_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_cohere_tiny.py")
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
	const ckpt = "../testdata/cohere-tiny"
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_cohere_tiny.py", ckpt)
	}
	m, err := Load(ckpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	// Last-token logits over the full prompt (sequential decode path).
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

// TestCohere_batchedPrefillMatchesSequential guards the OTHER parallel-block path:
// forwardN (the batched prefill/verify used for K>1) must reproduce the sequential
// decode's last-token logits EXACTLY (f64 accumulation is bit-identical by design),
// so the parallel branch added to runLayersFromEmbedN can't silently diverge from
// the streaming one. Independent of the HF golden — a pure internal consistency
// check that runs whenever the tiny checkpoint is present.
func TestCohere_batchedPrefillMatchesSequential(t *testing.T) {
	const ckpt = "../testdata/cohere-tiny"
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_cohere_tiny.py", ckpt)
	}
	m, err := Load(ckpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	ids := []int{3, 141, 7, 88, 219, 14, 66, 190}

	// Sequential last-token logits.
	seqCache := m.NewCache(len(ids))
	for _, id := range ids[:len(ids)-1] {
		if _, err := m.runLayers(id, seqCache); err != nil {
			t.Fatalf("seq runLayers: %v", err)
		}
	}
	seq, err := m.forward(ids[len(ids)-1], seqCache)
	if err != nil {
		t.Fatalf("seq forward: %v", err)
	}

	// Batched prefill: forwardN returns per-position logits; take the last row.
	rows, err := m.forwardN(context.Background(), ids, m.NewCache(len(ids)))
	if err != nil {
		t.Fatalf("batched forwardN: %v", err)
	}
	batchLogits := rows[len(rows)-1]
	if len(batchLogits) != len(seq) {
		t.Fatalf("batched %d logits, sequential %d", len(batchLogits), len(seq))
	}
	var maxAbs float64
	for i := range seq {
		if d := math.Abs(float64(seq[i]) - float64(batchLogits[i])); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("batched vs sequential maxAbs=%.3e", maxAbs)
	if maxAbs > 1e-4 {
		t.Errorf("batched prefill diverged from sequential: maxAbs %.3e > 1e-4", maxAbs)
	}
}

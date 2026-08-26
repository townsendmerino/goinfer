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

// TestCohere2_forwardParity is the Phase-2 end-to-end gate for Cohere2 / Command-R7B
// (docs/task-model-family-cohere.md). It drives goinfer's whole cohere2 forward —
// Load of the tiny safetensors checkpoint (generic loader, cohereTensorSchema) into
// the NormParallel + interleaved-sliding-window + per-layer-NoPE path — against the
// HF Cohere2ForCausalLM oracle in scripts/pin_cohere2_tiny.py. f32, so it's a TIGHT
// gate: last-token argmax exact, full-vocab cosine, greedy continuation identical.
//
// Beyond cohere1's primitives it pins: the sliding-window layers (prompt > window),
// and NoPE on the global layer (only sliding layers carry RoPE).
//
// Regenerate:  python3 scripts/pin_cohere2_tiny.py
func TestCohere2_forwardParity(t *testing.T) {
	raw, err := os.ReadFile("../testdata/cohere2_tiny_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_cohere2_tiny.py")
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
	const ckpt = "../testdata/cohere2-tiny"
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_cohere2_tiny.py", ckpt)
	}
	m, err := Load(ckpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	a := m.w.arch
	if a.Name != "cohere2" {
		t.Fatalf("arch = %q, want cohere2", a.Name)
	}
	// The interleave itself: pattern=4 over 4 layers ⇒ layers 0-2 sliding+RoPE, layer
	// 3 global+NoPE. A misclassification silently corrupts logits, so assert it.
	for l := 0; l < a.NumLayers; l++ {
		wantGlobal := (l+1)%4 == 0
		if a.isGlobalLayer(l) != wantGlobal {
			t.Errorf("layer %d isGlobal=%v, want %v", l, a.isGlobalLayer(l), wantGlobal)
		}
		if a.isNoPELayer(l) != wantGlobal { // global == NoPE in cohere2
			t.Errorf("layer %d isNoPE=%v, want %v", l, a.isNoPELayer(l), wantGlobal)
		}
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
	t.Logf("argmax: got %d | want %d", got, g.Argmax)
	if got != g.Argmax {
		t.Errorf("argmax = %d, want %d", got, g.Argmax)
	}

	var dot, na, nb, maxAbs float64
	for i, want := range g.LastLogits {
		aa := float64(logits[i])
		if d := math.Abs(aa - want); d > maxAbs {
			maxAbs = d
		}
		dot += aa * want
		na += aa * aa
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

// TestCohere2_batchedPrefillMatchesSequential guards the batched forwardN parallel +
// NoPE + sliding path against the sequential decode (bit-identical by design), so the
// prefill can't silently diverge on the sliding/NoPE interleave.
func TestCohere2_batchedPrefillMatchesSequential(t *testing.T) {
	const ckpt = "../testdata/cohere2-tiny"
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_cohere2_tiny.py", ckpt)
	}
	m, err := Load(ckpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	ids := []int{3, 141, 7, 88, 219, 14, 66, 190, 42, 5, 133, 201, 17, 250, 99, 60}

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

	rows, err := m.forwardN(context.Background(), ids, m.NewCache(len(ids)))
	if err != nil {
		t.Fatalf("batched forwardN: %v", err)
	}
	batch := rows[len(rows)-1]
	if len(batch) != len(seq) {
		t.Fatalf("batched %d logits, sequential %d", len(batch), len(seq))
	}
	var maxAbs float64
	for i := range seq {
		if d := math.Abs(float64(seq[i]) - float64(batch[i])); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("batched vs sequential maxAbs=%.3e", maxAbs)
	if maxAbs > 1e-4 {
		t.Errorf("batched prefill diverged from sequential: maxAbs %.3e > 1e-4", maxAbs)
	}
}

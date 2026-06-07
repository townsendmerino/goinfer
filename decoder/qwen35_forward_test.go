package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestQwen35_forwardParity loads the real tiny qwen3_5_moe checkpoint (saved by
// scripts/pin_qwen35_forward.py) through goinfer's actual loader + forward path
// and asserts both the last-position argmax/logit-cosine and the greedy decode
// match the HF golden — the end-to-end gate for the hybrid (Gated DeltaNet linear
// layers + gated-softmax layers + MoE).
func TestQwen35_forwardParity(t *testing.T) {
	const golden = "../testdata/qwen35_forward_golden.json"
	const ckpt = "../testdata/qwen35-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden/checkpoint — run scripts/pin_qwen35_forward.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_qwen35_forward.py", ckpt)
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

	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	if m.w.arch.qwen35 == nil {
		t.Fatalf("arch = %q, want qwen3_5_moe", m.w.arch.Name)
	}

	// 1. Prefill the prompt; check the last-position logits vs HF.
	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	for _, id := range g.PromptIDs[:len(g.PromptIDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.PromptIDs[len(g.PromptIDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if argmax(logits) != g.Argmax {
		t.Errorf("prefill argmax = %d, want %d", argmax(logits), g.Argmax)
	}
	var dot, na, nb float64
	for i := range logits {
		got, want := float64(logits[i]), float64(g.LastLogits[i])
		dot += got * want
		na += got * got
		nb += want * want
	}
	cos := dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
	t.Logf("qwen3_5_moe prefill: argmax=%d cosine=%.8f", argmax(logits), cos)
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.8f < 0.9999", cos)
	}

	// 2. Greedy decode off the same cache; must match HF id-for-id.
	got := make([]int, 0, g.NNew)
	for i := 0; i < g.NNew; i++ {
		next := argmax(logits)
		got = append(got, next)
		if i < g.NNew-1 {
			if logits, err = m.forward(next, cache); err != nil {
				t.Fatalf("decode forward %d: %v", i, err)
			}
		}
	}
	for i := range got {
		if got[i] != g.ContinuationIDs[i] {
			t.Fatalf("continuation diverges at %d: got %v, want %v", i, got, g.ContinuationIDs)
		}
	}
	t.Logf("greedy continuation matches HF: %v", got)
}

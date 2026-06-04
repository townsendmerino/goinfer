package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// Llama-adapter forward parity. Loads a real Llama checkpoint (TinyLlama-1.1B,
// Llama-2 architecture: RMS no-offset, Pre2, SwiGLU, single-base RoPE, NO
// QK-norm, untied head, head_dim derived from hidden/heads) through the SAME
// generic forward pass and checks next-token logits match the HF float32
// oracle. Like the Qwen3 test it feeds the golden's HF token ids to isolate the
// forward pass from the tokenizer.
//
// Regenerate:  .venv/bin/python scripts/pin_llama_forward.py
const (
	llamaModelDir        = "../testdata/tinyllama-1.1b"
	llamaForwardGolden   = "../testdata/llama_forward_golden.json"
	llamaForwardFullPath = "../testdata/llama_forward_full.json"
)

func TestLlama_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs TinyLlama-1.1B")
	}
	raw, err := os.ReadFile(llamaForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Llama golden at %s — regenerate with scripts/pin_llama_forward.py", llamaForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden // same shape as the Gemma/Qwen3 forward golden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(llamaModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Llama checkpoint at %s", llamaModelDir)
	}

	m, err := Load(llamaModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "llama" {
		t.Fatalf("resolved arch %q, want llama", m.w.arch.Name)
	}
	if m.w.arch.TiedLMHead {
		t.Errorf("expected untied LM head for TinyLlama (lm_head.weight present)")
	}
	if m.w.arch.QKNorm {
		t.Errorf("Llama must not use QK-norm")
	}
	if m.w.arch.HeadDim != 64 { // derived: hidden 2048 / 32 heads
		t.Errorf("HeadDim = %d, want 64 (derived from hidden/heads)", m.w.arch.HeadDim)
	}

	cache := m.NewCache(len(g.IDs))
	for _, id := range g.IDs[:len(g.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.IDs[len(g.IDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(logits) != g.Vocab {
		t.Fatalf("got %d logits, want vocab %d", len(logits), g.Vocab)
	}

	if got := argmax(logits); got != g.Argmax {
		t.Errorf("argmax = %d, want %d (logit[got]=%.4f logit[want]=%.4f)",
			got, g.Argmax, logits[got], logits[g.Argmax])
	}

	const valTol = 5e-3
	var maxSampleΔ float64
	for _, kv := range g.Sample {
		id := int(kv[0])
		d := math.Abs(float64(logits[id]) - kv[1])
		if d > maxSampleΔ {
			maxSampleΔ = d
		}
		if d > valTol {
			t.Errorf("sample id=%d logit=%.5f want %.5f (Δ%.5f)", id, logits[id], kv[1], d)
		}
	}
	for r, kv := range g.TopK {
		id := int(kv[0])
		if d := math.Abs(float64(logits[id]) - kv[1]); d > valTol {
			t.Errorf("top_k[%d] id=%d logit=%.5f want %.5f (Δ%.5f)", r, id, logits[id], kv[1], d)
		}
	}

	cos := fullCosine(t, logits, llamaForwardFullPath)
	t.Logf("llama: argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v", argmax(logits), g.Argmax, maxSampleΔ, cos)
}

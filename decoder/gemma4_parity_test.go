package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestGemma4_logitParity is the Increment-3 gate: goinfer's gemma4 forward (int8
// quantized GGUF) vs the HF bf16 oracle. Argmax must match (the correctness
// gate); cosine over the sampled-256 logits must clear the quant-vs-bf16 bar.
// Skips without the golden or the GGUF asset.
func TestGemma4_logitParity(t *testing.T) {
	requireHeavyModel(t)
	raw, err := os.ReadFile("../testdata/gemma4_forward_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no gemma4 golden; run scripts/pin_gemma4_forward.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	path := os.Getenv("HOME") + "/models/gemma-4-E2B_q4_0-it.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no E2B gguf (%v)", err)
	}
	m, err := Load(path, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
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

	got := argmax(logits)
	t.Logf("argmax: got %d (logit %.4f) | want %d (golden logit %.4f)",
		got, logits[got], g.Argmax, logits[g.Argmax])
	if got != g.Argmax {
		t.Errorf("argmax = %d, want %d", got, g.Argmax)
	}

	// Cosine over the seeded 256-sample (quantized int8 vs bf16 oracle).
	var dot, na, nb float64
	for _, kv := range g.Sample {
		a := float64(logits[int(kv[0])])
		b := kv[1]
		dot += a * b
		na += a * a
		nb += b * b
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	t.Logf("sample-256 cosine (int8int8 vs bf16) = %.5f", cos)
	if cos < 0.98 {
		t.Errorf("sample cosine %.5f < 0.98", cos)
	}
}

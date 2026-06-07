package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestGemma4_12B_logitParity gates the Gemma 4 12B forward — the dense, PLE-free
// variant whose global layers use attention_k_eq_v (K=V) and proportional
// (partial-rotary) RoPE, the paths E2B/E4B never exercise. goinfer's quantized
// forward must match the HF bf16 oracle: argmax exact (the chat-templated golden's
// argmax is the first answer token, so this also gates coherence) + sample-256
// cosine over the quant floor. Skips without the golden or the GGUF asset.
func TestGemma4_12B_logitParity(t *testing.T) {
	raw, err := os.ReadFile("../testdata/gemma4_12b_forward_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no 12B golden; run scripts/pin_gemma4_12b.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	path := os.Getenv("HOME") + "/models/gemma-4-12b-gguf/gemma-4-12b-it-qat-q4_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 12B gguf (%v)", err)
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

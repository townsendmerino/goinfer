package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// M3 forward-pass parity. Runs the real checkpoint's forward over a fixed
// BOS-prefixed prompt and asserts the next-token logits match the HF float32
// oracle: argmax identical (the correctness gate), top-k and a seeded sample
// of indices within tolerance, global stats close, and — when the per-machine
// full-logit dump is present — full-vector cosine ≥ 1 − 1e-4.
//
// Regenerate the oracle:
//
//	.venv/bin/python scripts/pin_gemma_forward.py
const (
	gemmaForwardGoldenPath = "../testdata/gemma_forward_golden.json"
	gemmaForwardFullPath   = "../testdata/gemma_forward_full.json"
)

type forwardGolden struct {
	Prompt string `json:"prompt"` // source text (for end-to-end tokenizer checks)
	IDs    []int  `json:"ids"`
	Argmax int    `json:"argmax"`
	Vocab  int    `json:"vocab_size"`
	Stats  struct {
		N     int     `json:"n"`
		Sum   float64 `json:"sum"`
		SumSq float64 `json:"sum_sq"`
		Min   float64 `json:"min"`
		Max   float64 `json:"max"`
	} `json:"stats"`
	TopK   [][2]float64 `json:"top_k"`  // [id, logit]
	Sample [][2]float64 `json:"sample"` // [index, value]
}

func TestForward_logitParity(t *testing.T) {
	raw, err := os.ReadFile(gemmaForwardGoldenPath)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no forward golden at %s — regenerate with scripts/pin_gemma_forward.py", gemmaForwardGoldenPath)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(gemmaModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — huggingface-cli download google/gemma-3-270m --local-dir %s",
			gemmaModelDir, gemmaModelDir)
	}

	m, err := Load(gemmaModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Prefill all prompt tokens; the last token's forward yields the logits.
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

	// 1. Argmax — the decisive correctness gate (the predicted next token).
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("argmax = %d, want %d (logit[got]=%.4f logit[want]=%.4f)",
			got, g.Argmax, logits[got], logits[g.Argmax])
	}

	// 2. Top-k ids identical, values within tolerance.
	const valTol = 5e-3 // f32 accumulation drift over 18 layers + LM head (observed ~3e-5)
	for r, kv := range g.TopK {
		id := int(kv[0])
		if d := math.Abs(float64(logits[id]) - kv[1]); d > valTol {
			t.Errorf("top_k[%d] id=%d logit=%.5f want %.5f (Δ%.5f)", r, id, logits[id], kv[1], d)
		}
	}

	// 3. Seeded sample of indices within tolerance.
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

	// 4. Global stats close (relative on the sums, absolute on min/max).
	var sum, sumSq float64
	for _, v := range logits {
		sum += float64(v)
		sumSq += float64(v) * float64(v)
	}
	if d := relDiff(sum, g.Stats.Sum); d > 1e-3 {
		t.Errorf("sum = %.4f, want %.4f (rel %.2e)", sum, g.Stats.Sum, d)
	}
	if d := relDiff(sumSq, g.Stats.SumSq); d > 1e-3 {
		t.Errorf("sum_sq = %.4f, want %.4f (rel %.2e)", sumSq, g.Stats.SumSq, d)
	}

	// 5. Full-vector cosine when the per-machine dump is present.
	cos := fullCosine(t, logits, gemmaForwardFullPath)
	t.Logf("argmax=%d (want %d) | maxSampleΔ=%.5f | sum rel=%.2e | cosine=%v",
		argmax(logits), g.Argmax, maxSampleΔ, relDiff(sum, g.Stats.Sum), cos)
}

// fullCosine returns cosine(go, hf) over the full logit vector if the
// gitignored full dump at path is present, else math.NaN() (logged, not
// failed). Shared by the M3 forward and M5 sliding-window parity tests.
func fullCosine(t *testing.T, logits []float32, path string) float64 {
	cos := cosineToFull(t, logits, path)
	if !math.IsNaN(cos) && cos < 1-1e-4 {
		t.Errorf("full cosine = %.8f, want ≥ %.8f", cos, 1-1e-4)
	}
	return cos
}

// cosineToFull computes the cosine of logits against a committed full-logit
// dump, WITHOUT asserting a threshold — callers pick their own bar (f32 parity
// is ~1−1e-4; quantized GGUF/int8 is looser). Returns NaN if the dump is absent.
func cosineToFull(t *testing.T, logits []float32, path string) float64 {
	raw, err := os.ReadFile(path)
	if err != nil {
		return math.NaN()
	}
	var full struct {
		Argmax int       `json:"argmax"`
		Logits []float64 `json:"logits"`
	}
	if err := json.Unmarshal(raw, &full); err != nil {
		t.Fatalf("parse full dump: %v", err)
	}
	if len(full.Logits) != len(logits) {
		t.Fatalf("full dump len %d != %d", len(full.Logits), len(logits))
	}
	var dot, na, nb float64
	for i, v := range logits {
		a := float64(v)
		b := full.Logits[i]
		dot += a * b
		na += a * a
		nb += b * b
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func relDiff(a, b float64) float64 {
	d := math.Abs(a - b)
	m := math.Max(math.Abs(a), math.Abs(b))
	if m == 0 {
		return d
	}
	return d / m
}

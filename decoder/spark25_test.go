package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestSpark25Tiny_textParity is Gate 1 (docs/tasks/task-spark-x2-5.md): "loads and matches the
// tiny golden." Pins the new spark2_5 forward — fused QKV split, sigmoid head-wise attention-
// output gate, 1:3 sliding:full interleave with per-layer-type partial RoPE, and a GATED MLP
// using exact-erf GELU (gegluExact) — against a tiny-random HF oracle (the real
// modeling_spark.py, trust_remote_code, run under transformers==4.57.1 — see
// scripts/pin_spark2_5_tiny.py's own doc comment for why that exact version is required).
//
// The fixture deliberately decouples head_dim (32) from hidden_size/num_heads (64/4=16), matching
// the real checkpoint's own decoupling (head_dim 256, hidden/heads = 2560/16 = 160) — a loader
// that assumes head_dim=hidden/heads would silently read the wrong shape.
func TestSpark25Tiny_textParity(t *testing.T) {
	raw, err := os.ReadFile("testdata/spark2_5_tiny_text_golden.json")
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden — run scripts/pin_spark2_5_tiny.py under a transformers==4.57.1 venv")
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
	const ckpt = "testdata/spark2-5-tiny"
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no tiny checkpoint (%s) — run scripts/pin_spark2_5_tiny.py", ckpt)
	}
	m, err := Load(ckpt, Options{Quant: "f32"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if m.w.arch.Name != "spark2_5" {
		t.Fatalf("arch = %q, want spark2_5", m.w.arch.Name)
	}
	if m.w.arch.HeadDim != 32 {
		t.Fatalf("HeadDim = %d, want 32 (decoupled from hidden/heads=16 — a wrong value means the "+
			"loader silently fell back to hidden/heads instead of reading the explicit head_dim)", m.w.arch.HeadDim)
	}
	if m.w.arch.AttnGate != GateSigmoid {
		t.Fatalf("AttnGate = %v, want GateSigmoid", m.w.arch.AttnGate)
	}
	if m.w.arch.NormPlacement != NormPre2 {
		t.Fatalf("NormPlacement = %v, want NormPre2 (sequential) — Spark-X2.5 is NOT Cohere's "+
			"parallel block, despite the audit's original claim; see spark25Architecture's doc comment", m.w.arch.NormPlacement)
	}
	if m.w.arch.TiedLMHead {
		t.Fatal("fixture resolved as tied; it is built untied on purpose (tie_word_embeddings=False — " +
			"see pin_spark2_5_tiny.py's own note on why), so a tied read here means the config was misparsed")
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
	t.Logf("spark2-5-tiny parity: argmax=%d cosine=%.8f maxAbs=%.3e over %d vocab",
		got, cos, maxAbs, len(g.LastLogits))
	if cos < 0.9999 {
		t.Errorf("cosine %.8f < 0.9999", cos)
	}
	if maxAbs > 5e-2 {
		t.Errorf("maxAbs %.3e > 5e-2", maxAbs)
	}

	// A MATCHING ARGMAX MEANS LITTLE ON ITS OWN — both LFM2 bugs held argmax while the logit
	// cosine was 0.897. The greedy continuation is the part that compounds, and here it also
	// exercises the sliding-window clip: PROMPT (16 tokens) is longer than sliding_window (8).
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

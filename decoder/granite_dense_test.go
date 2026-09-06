package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// F3 dense Granite 4.2 parity (ibm-granite/granite-4.2-{3b,8b,30b}, model_type "granite"): a
// plain llama skeleton plus three of Granite's four scalar multipliers (embedding_multiplier,
// attention_multiplier, logits_scaling — all already generic on Architecture; residual_multiplier
// stays at its identity 1.0, the only value any released 4.2 size ships and the only value
// validateGraniteDense accepts). Loads a tiny seeded GraniteForCausalLM with NON-trivial
// multipliers through the generic forward and matches the HF float32 oracle, so a forward that
// drops any one of them fails parity.
//
// Regenerate (seeded tiny GraniteForCausalLM checkpoint + golden, both reproducible):
//
//	~/.venv-nemotron3/bin/python scripts/pin_granite_dense_tiny.py
const (
	graniteDenseModelDir        = "../testdata/granite-dense-tiny"
	graniteDenseForwardGolden   = "../testdata/granite_dense_forward_golden.json"
	graniteDenseForwardFullPath = "../testdata/granite_dense_forward_full.json"
)

func TestGraniteDense_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs Granite-dense-tiny")
	}
	raw, err := os.ReadFile(graniteDenseForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Granite-dense golden at %s — regenerate with scripts/pin_granite_dense_tiny.py", graniteDenseForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(graniteDenseModelDir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Granite-dense checkpoint at %s — regenerate with scripts/pin_granite_dense_tiny.py", graniteDenseModelDir)
	}

	m, err := Load(graniteDenseModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "granite" {
		t.Fatalf("resolved arch %q, want granite", m.w.arch.Name)
	}
	if m.w.arch.MoE != nil {
		t.Fatalf("expected no MoE (dense Granite)")
	}
	// The fixture's multipliers are all non-1.0 (except residual, pinned to 1.0) — a wrong or
	// dropped derivation would show up as a resolved value equal to the "unset" default instead.
	if got, want := m.w.arch.EmbedScale, 12.0; got != want {
		t.Errorf("EmbedScale = %v, want %v (embedding_multiplier)", got, want)
	}
	if got, want := m.w.arch.AttnScale, 0.5; got != want {
		t.Errorf("AttnScale = %v, want %v (attention_multiplier)", got, want)
	}
	if got, want := m.w.arch.LogitScale, 6.0; got != want {
		t.Errorf("LogitScale = %v, want %v (logits_scaling)", got, want)
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

	cos := fullCosine(t, logits, graniteDenseForwardFullPath)
	t.Logf("granite (dense): argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v",
		argmax(logits), g.Argmax, maxSampleΔ, cos)
	emitParityRow(t, "granite", "tiny-golden", "HF f32 (granite-dense-tiny seeded fixture, non-trivial scalar multipliers)", 100.0, cos, cos)
}

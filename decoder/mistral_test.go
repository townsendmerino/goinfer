package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// Mistral family parity. Mistral is the llama descriptor with sliding-window
// attention on every layer. The golden uses a prompt LONGER than TinyMistral's
// 32-token window, so the last position genuinely attends to only the trailing
// 32 tokens — matching HF here proves the all-layer sliding-window path, not
// just the llama-equivalent forward. Reuses Gemma's M5 window machinery.
//
// Regenerate:  PIN_PROMPT="<long text>" .venv/bin/python scripts/pin_llama_forward.py testdata/tinymistral-248m mistral
const (
	mistralModelDir        = "../testdata/tinymistral-248m"
	mistralForwardGolden   = "../testdata/mistral_forward_golden.json"
	mistralForwardFullPath = "../testdata/mistral_forward_full.json"
)

func TestMistral_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs TinyMistral-248M")
	}
	raw, err := os.ReadFile(mistralForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Mistral golden at %s — regenerate with scripts/pin_llama_forward.py", mistralForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(mistralModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Mistral checkpoint at %s", mistralModelDir)
	}

	m, err := Load(mistralModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "mistral" {
		t.Fatalf("resolved arch %q, want mistral", m.w.arch.Name)
	}
	if m.w.arch.SlidingWindow <= 0 {
		t.Fatalf("expected a sliding window, got %d", m.w.arch.SlidingWindow)
	}
	// Every layer must be local (the all-local Mistral pattern), unlike Gemma's 5:1.
	for i := 0; i < m.w.arch.NumLayers; i++ {
		if m.w.arch.isGlobalLayer(i) {
			t.Fatalf("layer %d is global; Mistral is all-local", i)
		}
	}
	// The test is only meaningful if the prompt exceeds the window.
	if len(g.IDs) <= m.w.arch.SlidingWindow {
		t.Fatalf("prompt (%d ids) must exceed the window (%d) to exercise sliding attention", len(g.IDs), m.w.arch.SlidingWindow)
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

	cos := fullCosine(t, logits, mistralForwardFullPath)
	t.Logf("mistral: %d-token prompt (window %d) | argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v",
		len(g.IDs), m.w.arch.SlidingWindow, argmax(logits), g.Argmax, maxSampleΔ, cos)
}

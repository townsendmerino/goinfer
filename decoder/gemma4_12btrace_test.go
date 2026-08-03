package decoder

import (
	"encoding/json"
	"os"
	"strconv"
	"testing"
)

// TestGemma4_12B_trace drives the 12B GGUF over the HF trace's prompt and dumps a
// last-position per-layer residual trace + the layer 4/5 q/k/v, for an offline
// cosine diff against the HF reference (scripts/diff_gemma4_12b.py). Debug aid for
// the parked K=V forward — gated on G4_TRACE so it never runs in normal CI.
//
//	G4_ALLOW_KV=1 G4_TRACE=1 go test ./decoder/ -run TestGemma4_12B_trace -v
func TestGemma4_12B_trace(t *testing.T) {
	requireHeavyModel(t)
	if os.Getenv("G4_TRACE") == "" {
		t.Skip("set G4_TRACE=1 to run the 12B trace dump")
	}
	hfRaw, err := os.ReadFile("../testdata/gemma4_12b_trace.json")
	if err != nil {
		t.Skipf("no HF trace: %v (run scripts/dump_gemma4_12b_trace.py)", err)
	}
	var hf struct {
		IDs    []int `json:"ids"`
		Argmax int   `json:"argmax"`
	}
	if err := json.Unmarshal(hfRaw, &hf); err != nil {
		t.Fatalf("parse HF trace: %v", err)
	}
	path := os.Getenv("HOME") + "/models/gemma-4-12b-gguf/gemma-4-12b-it-qat-q4_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 12B gguf (%v)", err)
	}
	m, err := Load(path, Options{Quant: ""}) // f32 weights (no activation int8 noise)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	hidden := map[string][]float32{}
	qkv := map[string][]float32{}

	cache := m.NewCache(len(hf.IDs))
	for _, id := range hf.IDs[:len(hf.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	// Trace only the final token's forward.
	g4traceHidden = func(layer int, h []float32) {
		cp := make([]float32, len(h))
		copy(cp, h)
		hidden[strconv.Itoa(layer)] = cp
	}
	g4traceQKV = func(layer int, q, k, v []float32) {
		if layer != 4 && layer != 5 {
			return
		}
		store := func(name string, x []float32) {
			if x == nil {
				return
			}
			cp := make([]float32, len(x))
			copy(cp, x)
			qkv[strconv.Itoa(layer)+"."+name] = cp
		}
		store("q", q)
		store("k", k)
		store("v", v)
	}
	defer func() { g4traceHidden = nil; g4traceQKV = nil }()

	logits, err := m.forward(hf.IDs[len(hf.IDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	got := argmax(logits)
	t.Logf("goinfer 12B argmax = %d (logit %.4f) | HF argmax = %d", got, logits[got], hf.Argmax)
	if got == hf.Argmax {
		t.Logf("ARGMAX MATCHES HF — 12B forward looks correct!")
	}

	out := map[string]any{
		"argmax":      got,
		"argmax_ok":   got == hf.Argmax,
		"hidden_last": hidden,
		"qkv":         qkv,
	}
	f, err := os.Create("../testdata/gemma4_12b_goinfer_trace.json")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer f.Close()
	if err := json.NewEncoder(f).Encode(out); err != nil {
		t.Fatalf("encode: %v", err)
	}
	t.Logf("wrote ../testdata/gemma4_12b_goinfer_trace.json (%d layers, %d qkv probes)", len(hidden), len(qkv))
}

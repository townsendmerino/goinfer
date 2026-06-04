package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// G6 Mixtral / MoE parity. Mixtral is the llama descriptor with the dense FFN
// replaced by a sparse mixture of experts: a router softmaxes over all experts,
// the top-k run as SwiGLU MLPs, and their outputs combine weighted by the
// renormalized router probabilities. Loads real Mixtral-tiny (8 experts, top-2)
// through the generic forward and matches the HF float32 oracle — exercising
// the router, top-k selection, and weighted expert combine.
//
// Regenerate:  .venv/bin/python scripts/pin_llama_forward.py testdata/mixtral-tiny mixtral
const (
	mixtralModelDir        = "../testdata/mixtral-tiny"
	mixtralForwardGolden   = "../testdata/mixtral_forward_golden.json"
	mixtralForwardFullPath = "../testdata/mixtral_forward_full.json"
)

func TestMixtral_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs Mixtral-tiny")
	}
	raw, err := os.ReadFile(mixtralForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Mixtral golden at %s — regenerate with scripts/pin_llama_forward.py", mixtralForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(mixtralModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Mixtral checkpoint at %s", mixtralModelDir)
	}

	m, err := Load(mixtralModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "mixtral" {
		t.Fatalf("resolved arch %q, want mixtral", m.w.arch.Name)
	}
	if m.w.arch.MoE == nil {
		t.Fatalf("expected a MoE config")
	}
	if m.w.arch.MoE.NumExperts != 8 || m.w.arch.MoE.TopK != 2 {
		t.Errorf("MoE = %dx top-%d, want 8x top-2", m.w.arch.MoE.NumExperts, m.w.arch.MoE.TopK)
	}
	if got := len(m.w.Layers[0].Experts); got != 8 {
		t.Fatalf("layer 0 loaded %d experts, want 8", got)
	}
	if m.w.Layers[0].Router.rows != 8 {
		t.Fatalf("router rows = %d, want 8", m.w.Layers[0].Router.rows)
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

	cos := fullCosine(t, logits, mixtralForwardFullPath)
	t.Logf("mixtral: %dx top-%d | argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v",
		m.w.arch.MoE.NumExperts, m.w.arch.MoE.TopK, argmax(logits), g.Argmax, maxSampleΔ, cos)
}

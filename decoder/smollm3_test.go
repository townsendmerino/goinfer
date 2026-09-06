package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// G4 (docs/task-families-2026-09.md, batch 2) SmolLM3-3B parity: a plain llama-shaped dense GQA
// model with per-layer NoPE via `no_rope_layers` — a field whose VALUES are the opposite of what
// its name suggests (1 = HAS rope, 0 = NoPE, verified against the real modeling_smollm3.py), on
// every 4th layer. Reuses the SAME Config field/convention llama4_text already established and
// the SAME layerNoPE Architecture hook cohere2Architecture already populates.
//
// Regenerate (seeded tiny SmolLM3ForCausalLM checkpoint + golden, both reproducible):
//
//	~/.venv-nemotron3/bin/python scripts/pin_smollm3_tiny.py
const (
	smollm3ModelDir        = "../testdata/smollm3-tiny"
	smollm3ForwardGolden   = "../testdata/smollm3_forward_golden.json"
	smollm3ForwardFullPath = "../testdata/smollm3_forward_full.json"
)

func TestSmolLM3_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs SmolLM3-tiny")
	}
	raw, err := os.ReadFile(smollm3ForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no SmolLM3 golden at %s — regenerate with scripts/pin_smollm3_tiny.py", smollm3ForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(smollm3ModelDir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no SmolLM3 checkpoint at %s — regenerate with scripts/pin_smollm3_tiny.py", smollm3ModelDir)
	}

	m, err := Load(smollm3ModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "smollm3" {
		t.Fatalf("resolved arch %q, want smollm3", m.w.arch.Name)
	}
	// The fixture's own no_rope_layers is [1,1,1,0] (4 layers): only the LAST layer is NoPE.
	// Getting the field's polarity backwards would flip this exactly (3 NoPE, 1 RoPE).
	wantNoPE := []bool{false, false, false, true}
	for i, want := range wantNoPE {
		if got := m.w.arch.isNoPELayer(i); got != want {
			t.Errorf("isNoPELayer(%d) = %v, want %v (no_rope_layers=[1,1,1,0]: 1=has-rope, 0=NoPE)", i, got, want)
		}
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

	cos := fullCosine(t, logits, smollm3ForwardFullPath)
	t.Logf("smollm3: argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v",
		argmax(logits), g.Argmax, maxSampleΔ, cos)
	emitParityRow(t, "smollm3", "tiny-golden", "HF f32 (smollm3-tiny seeded fixture, per-layer NoPE on layer 3)", 100.0, cos, cos)
}

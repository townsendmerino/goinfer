package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// Qwen2 family parity. Qwen2/Qwen2.5 is the llama descriptor plus q/k/v
// projection bias (QKVBias) — the one knob this milestone adds. Loads the real
// Qwen2.5-0.5B-Instruct through the generic forward and checks next-token logits
// match the HF float32 oracle. Also an end-to-end check that the Go byte-level
// tokenizer reproduces the HF ids (Qwen2 prepends no BOS).
//
// Regenerate:  .venv/bin/python scripts/pin_llama_forward.py testdata/qwen2.5-0.5b qwen2
const (
	qwen2ModelDir        = "../testdata/qwen2.5-0.5b"
	qwen2ForwardGolden   = "../testdata/qwen2_forward_golden.json"
	qwen2ForwardFullPath = "../testdata/qwen2_forward_full.json"
)

func TestQwen2_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs Qwen2.5-0.5B")
	}
	raw, err := os.ReadFile(qwen2ForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Qwen2 golden at %s — regenerate with scripts/pin_llama_forward.py", qwen2ForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(qwen2ModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Qwen2 checkpoint at %s", qwen2ModelDir)
	}

	// End-to-end tokenizer check (Qwen2 adds no BOS, so addBOS is a no-op here).
	if tk, terr := tokenizer.Load(qwen2ModelDir); terr == nil {
		ids, eerr := tk.Encode(g.Prompt, true)
		if eerr != nil {
			t.Fatalf("tokenizer Encode: %v", eerr)
		}
		if !intsEqual(ids, g.IDs) {
			t.Errorf("Go tokenizer ids = %v, want HF ids %v", ids, g.IDs)
		}
	} else {
		t.Logf("tokenizer load skipped: %v", terr)
	}

	m, err := Load(qwen2ModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "qwen2" {
		t.Fatalf("resolved arch %q, want qwen2", m.w.arch.Name)
	}
	if !m.w.arch.QKVBias {
		t.Errorf("expected QKVBias=true for Qwen2")
	}
	if m.w.arch.QKNorm {
		t.Errorf("Qwen2 must not use QK-norm (that's Qwen3)")
	}
	// Sanity: the biases actually loaded.
	if m.w.Layers[0].QBias == nil || m.w.Layers[0].KBias == nil || m.w.Layers[0].VBias == nil {
		t.Fatalf("q/k/v bias not loaded for layer 0")
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

	cos := fullCosine(t, logits, qwen2ForwardFullPath)
	t.Logf("qwen2: argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v", argmax(logits), g.Argmax, maxSampleΔ, cos)
}

package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// G2 Qwen3 forward parity. Loads the real Qwen3-1.7B (a different family —
// 2-norm Pre2, RMS no-offset, SwiGLU, single-base RoPE, untied head, QK-norm)
// through the SAME generic forward pass and checks the next-token logits match
// the HF float32 oracle. The Go byte-level BPE tokenizer is G3, so the test
// feeds the golden's HF token ids to isolate the forward pass.
//
// Regenerate:  .venv/bin/python scripts/pin_qwen3_forward.py
const (
	qwen3ModelDir        = "../testdata/qwen3-1.7b"
	qwen3ForwardGolden   = "../testdata/qwen3_forward_golden.json"
	qwen3ForwardFullPath = "../testdata/qwen3_forward_full.json"
)

func TestQwen3_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs Qwen3-1.7B")
	}
	raw, err := os.ReadFile(qwen3ForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Qwen3 golden at %s — regenerate with scripts/pin_qwen3_forward.py", qwen3ForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden // same shape as the Gemma forward golden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(qwen3ModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Qwen3 checkpoint at %s", qwen3ModelDir)
	}

	m, err := Load(qwen3ModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "qwen3" {
		t.Fatalf("resolved arch %q, want qwen3", m.w.arch.Name)
	}
	if m.w.arch.TiedLMHead {
		t.Errorf("expected untied LM head for Qwen3-1.7B (lm_head.weight present)")
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

	cos := fullCosine(t, logits, qwen3ForwardFullPath)
	t.Logf("qwen3: argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v", argmax(logits), g.Argmax, maxSampleΔ, cos)
}

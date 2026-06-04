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

// G4 RoPE-scaling parity. Llama-3.2-1B uses llama3 rope_scaling (factor 32),
// which the descriptor bakes into the inv-freq table. This is also the first
// end-to-end check of the FULL Llama-3 pure-Go path: the byte-level BPE
// tokenizer encodes the prompt, then the generic forward (with llama3-scaled
// RoPE, derived head_dim, tied head) reproduces HF's next-token logits.
//
// Regenerate:  .venv/bin/python scripts/pin_llama_forward.py testdata/llama3.2-1b llama32
const (
	llama32ModelDir        = "../testdata/llama3.2-1b"
	llama32ForwardGolden   = "../testdata/llama32_forward_golden.json"
	llama32ForwardFullPath = "../testdata/llama32_forward_full.json"
)

func TestLlama32_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs Llama-3.2-1B")
	}
	raw, err := os.ReadFile(llama32ForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Llama-3.2 golden at %s — regenerate with scripts/pin_llama_forward.py", llama32ForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(llama32ModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Llama-3.2 checkpoint at %s", llama32ModelDir)
	}

	// End-to-end tokenizer check: the Go byte-level BPE must produce the same
	// ids HF did (addBOS=true → the <|begin_of_text|> the golden starts with).
	if tk, terr := tokenizer.Load(llama32ModelDir); terr == nil {
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

	m, err := Load(llama32ModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "llama" {
		t.Fatalf("resolved arch %q, want llama", m.w.arch.Name)
	}
	if m.w.arch.ropeScaling == nil || m.w.arch.ropeScaling.kind != ropeScaleLlama3 {
		t.Fatalf("expected llama3 rope scaling, got %+v", m.w.arch.ropeScaling)
	}
	if !m.w.arch.TiedLMHead {
		t.Errorf("expected tied LM head for Llama-3.2-1B")
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

	cos := fullCosine(t, logits, llama32ForwardFullPath)
	t.Logf("llama3.2: argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v", argmax(logits), g.Argmax, maxSampleΔ, cos)
}

func intsEqual(a []int, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

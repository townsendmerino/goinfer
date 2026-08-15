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

// G5 GPT-2 parity. GPT-2 is the first family that breaks the Llama mold:
// LayerNorm (mean-centered, with bias) instead of RMSNorm, learned absolute
// position embeddings instead of RoPE, a non-gated GELU MLP, fused q/k/v with
// bias, an attention output bias, and Conv1D ([in,out]) weight layout. Loads
// real GPT-2 small through the generic forward (+ the dedicated GPT-2 loader)
// and matches the HF float32 oracle. Also checks the Go byte-level tokenizer
// reproduces GPT-2's ids (GPT-2 prepends no BOS).
//
// Regenerate:  ~/.venv-vl/bin/python scripts/pin_gpt2_real.py
// (that script writes both the committed golden and the gitignored full-logit dump the
// cosine reads; the previously-named pin_llama_forward.py no longer exists.)
const (
	gpt2ModelDir        = "../testdata/gpt2"
	gpt2ForwardGolden   = "../testdata/gpt2_forward_golden.json"
	gpt2ForwardFullPath = "../testdata/gpt2_forward_full.json"
)

func TestGPT2_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs GPT-2 small")
	}
	raw, err := os.ReadFile(gpt2ForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no GPT-2 golden at %s — regenerate with scripts/pin_llama_forward.py", gpt2ForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(gpt2ModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no GPT-2 checkpoint at %s", gpt2ModelDir)
	}

	// End-to-end tokenizer check (GPT-2 adds no BOS).
	if tk, terr := tokenizer.Load(gpt2ModelDir); terr == nil {
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

	m, err := Load(gpt2ModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "gpt2" {
		t.Fatalf("resolved arch %q, want gpt2", m.w.arch.Name)
	}
	if m.w.arch.Norm != NormLayer {
		t.Errorf("expected LayerNorm")
	}
	if !m.w.arch.LearnedPosEmbed || !m.w.arch.NonGatedMLP || !m.w.arch.QKVBias || !m.w.arch.OutBias {
		t.Errorf("GPT-2 knobs wrong: learnedPos=%v nonGated=%v qkvBias=%v outBias=%v",
			m.w.arch.LearnedPosEmbed, m.w.arch.NonGatedMLP, m.w.arch.QKVBias, m.w.arch.OutBias)
	}
	if !m.w.arch.TiedLMHead {
		t.Errorf("expected tied LM head (wte) for GPT-2")
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

	cos := fullCosine(t, logits, gpt2ForwardFullPath)
	t.Logf("gpt2: argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v", argmax(logits), g.Argmax, maxSampleΔ, cos)

	// Record the measured row (no-op unless GOINFER_MANIFEST_EMIT; skipped if anything above
	// failed). Guarded on a real cosine: without the gitignored full-logit dump, fullCosine
	// returns NaN and there is no cosine to record — the manifest must not bank a metric the
	// run did not measure. Regenerate the dump with scripts/pin_gpt2_real.py.
	if !math.IsNaN(cos) {
		emitParityRow(t, "gpt2", "full-forward-oracle", "HF f32 (GPT-2 small, 124M)", 100.0, cos, cos)
	}
}

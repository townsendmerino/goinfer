package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestPhi3_descriptor checks the phi3 adapter resolves the llama-like descriptor: RMSNorm
// no-offset, Pre2, SwiGLU, no QK-norm, no bias, full rotary (no partial_rotary_factor),
// single-base RoPE, untied head finalized at load.
func TestPhi3_descriptor(t *testing.T) {
	cfg := &Config{
		ModelType: "phi3", HiddenDim: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2,
		VocabSize: 128, IntermediateDim: 128, RMSNormEps: 1e-5, RoPEGlobalBase: 10000,
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if schema != &phi3TensorSchema {
		t.Fatalf("schema mismatch")
	}
	if arch.Name != "phi3" || arch.Norm != NormRMS || arch.RMSAddOne || arch.NormPlacement != NormPre2 {
		t.Errorf("descriptor wrong: %+v", arch)
	}
	if arch.QKNorm || arch.QKVBias || arch.MoE != nil {
		t.Errorf("phi3 dense: QKNorm=%v QKVBias=%v MoE=%v", arch.QKNorm, arch.QKVBias, arch.MoE != nil)
	}
	if arch.HeadDim != 16 || arch.RotaryDim != 0 { // 64/4=16; full rotary
		t.Errorf("HeadDim=%d RotaryDim=%d, want 16/0", arch.HeadDim, arch.RotaryDim)
	}
}

// TestPhi3_textParity loads the tiny-random Phi-3 checkpoint and gates the full forward
// (last-logit cosine + greedy continuation, both m.forward and Generate) against the HF
// golden. Exercises the fused qkv_proj / gate_up_proj split + GQA + the generic llama
// forward. Fixture: scripts/pin_phi3_tiny.py.
func TestPhi3_textParity(t *testing.T) {
	const golden = "../testdata/phi3_tiny_text_golden.json"
	const ckpt = "../testdata/phi3-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_phi3_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_phi3_tiny.py", ckpt)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("phi3 text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999", cos)
	}

	got := make([]int, 0, g.NNew)
	for range g.NNew {
		id := argmax(logits)
		got = append(got, id)
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
	for i := range g.ContinuationIDs {
		if got[i] != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d (got %v want %v)", i, got[i], g.ContinuationIDs[i], got, g.ContinuationIDs)
			break
		}
	}

	// Generate() path (greedy, batched prefill) must match too.
	gm, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	defer gm.Close()
	out, _ := gm.Generate(context.Background(), g.PromptIDs, g.NNew, SamplingParams{})
	gen := make([]int, 0, g.NNew)
	for id := range out {
		gen = append(gen, id)
	}
	for i := range g.ContinuationIDs {
		if i >= len(gen) || gen[i] != g.ContinuationIDs[i] {
			t.Errorf("Generate continuation[%d] mismatch: got %v want %v", i, gen, g.ContinuationIDs)
			break
		}
	}
}

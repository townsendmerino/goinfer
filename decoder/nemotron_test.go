package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestNemotron_descriptor checks the nemotron_h adapter resolves the single-op-block
// seams from config alone: per-layer kind (mamba | attention | mlp), no RoPE, relu²
// non-gated MLP, plain RMSNorm, no multipliers.
func TestNemotron_descriptor(t *testing.T) {
	cfg := &Config{
		ModelType: "nemotron_h", HiddenDim: 64, NumLayers: 5, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		VocabSize: 256, IntermediateDim: 128, LayerNormEpsilon: 1e-5,
		MambaNumHeads: 8, MambaHeadDim: 8, SSMStateSize: 16, NGroups: 1, ConvKernel: 4,
		LayersBlockType: []string{"mamba", "attention", "mlp", "mamba", "attention"},
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if schema != &nemotronTensorSchema || arch.nemotron == nil {
		t.Fatalf("schema/nemotron mismatch")
	}
	want := []uint8{nemoMamba, nemoAttn, nemoMLP, nemoMamba, nemoAttn}
	for i, k := range want {
		if arch.nemotron.blockKind[i] != k {
			t.Errorf("blockKind[%d]=%d, want %d", i, arch.nemotron.blockKind[i], k)
		}
	}
	if arch.Act != ActReLU2 || !arch.NonGatedMLP {
		t.Errorf("Act=%v NonGatedMLP=%v, want ReLU2/true", arch.Act, arch.NonGatedMLP)
	}
	if arch.RotaryDim != 0 || arch.QKNorm || arch.QKVBias {
		t.Errorf("NoPE/plain attention expected: RotaryDim=%d QKNorm=%v QKVBias=%v", arch.RotaryDim, arch.QKNorm, arch.QKVBias)
	}
	if arch.LogitScale != 0 {
		t.Errorf("Nemotron has no logit scaling, got %v", arch.LogitScale)
	}
}

// TestNemotron_textParity loads the tiny-random NemotronH checkpoint and gates the
// full single-op-block forward (last-logit cosine + greedy continuation, both
// m.forward and Generate) against the HF golden. Exercises all three block kinds
// (mamba, NoPE attention, relu² MLP). Fixture: scripts/pin_nemotron_tiny.py.
func TestNemotron_textParity(t *testing.T) {
	const golden = "../testdata/nemotron_tiny_text_golden.json"
	const ckpt = "../testdata/nemotron-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_nemotron_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_nemotron_tiny.py", ckpt)
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
	t.Logf("nemotron text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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

	// Generate() path (greedy) must match too — guards the canBatchN exclusion.
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

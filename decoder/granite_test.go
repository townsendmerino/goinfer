package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestGranite_descriptor checks the granitemoehybrid adapter resolves the Granite
// seams from config alone: per-layer mamba/attention split, the four multipliers,
// MoE-on-every-layer with an ungated shared expert, attention scale = multiplier.
func TestGranite_descriptor(t *testing.T) {
	cfg := &Config{
		ModelType: "granitemoehybrid", HiddenDim: 64, NumLayers: 4, NumHeads: 4, NumKVHeads: 2,
		VocabSize: 256, IntermediateDim: 128, RMSNormEps: 1e-6,
		MambaNHeads: 8, MambaDHead: 16, MambaDState: 16, MambaDConv: 4, MambaNGroups: 1,
		NumLocalExperts: 8, NumExpertsPerTok: 2, SharedIntermediateSize: 128,
		LayerTypes:          []string{"mamba", "attention", "mamba", "mamba"},
		EmbeddingMultiplier: 12.0, AttentionMultiplier: 0.5, ResidualMultiplier: 0.22, LogitsScaling: 6.0,
		RopeParameters: json.RawMessage(`{"rope_type":"default","rope_theta":10000.0}`),
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if schema != &graniteTensorSchema {
		t.Fatalf("schema mismatch")
	}
	if !arch.isMambaLayer(0) || arch.isMambaLayer(1) || !arch.isMambaLayer(2) {
		t.Errorf("layer kinds wrong: L0=%v L1=%v L2=%v", arch.isMambaLayer(0), arch.isMambaLayer(1), arch.isMambaLayer(2))
	}
	if arch.granite.EmbMul != 12 || arch.granite.ResidMul != 0.22 {
		t.Errorf("multipliers EmbMul=%v ResidMul=%v", arch.granite.EmbMul, arch.granite.ResidMul)
	}
	if arch.AttnScale != 0.5 {
		t.Errorf("AttnScale=%v, want 0.5 (attention_multiplier)", arch.AttnScale)
	}
	if arch.LogitScale != 6 {
		t.Errorf("LogitScale=%v, want 6", arch.LogitScale)
	}
	if arch.MoE == nil || !arch.MoE.SharedUngated || !arch.MoE.NormTopKProb {
		t.Errorf("MoE config wrong: %+v", arch.MoE)
	}
}

// TestGranite_textParity loads the tiny-random GraniteMoeHybrid checkpoint and gates
// the full hybrid forward — last-logit cosine + greedy continuation — against the HF
// golden. Exercises every Granite seam at once: mamba + attention layers, MoE on
// every layer, and all four non-trivial multipliers. Fixture: scripts/pin_granite_tiny.py.
func TestGranite_textParity(t *testing.T) {
	const golden = "../testdata/granite_tiny_text_golden.json"
	const ckpt = "../testdata/granite-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_granite_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_granite_tiny.py", ckpt)
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
	t.Logf("granite text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
}

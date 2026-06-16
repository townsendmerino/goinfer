package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestLlama4_descriptor checks the llama4_text adapter resolves the iRoPE seams from config:
// per-layer RoPE/NoPE (no_rope_layers), per-layer dense/MoE (moe_layers), top-1 sigmoid
// routing + ungated shared expert, parameter-free L2 QK-norm flag, attn-temperature params,
// and the dense-vs-expert width split (intermediate_size_mlp vs intermediate_size).
func TestLlama4_descriptor(t *testing.T) {
	cfg := &Config{
		ModelType: "llama4_text", HiddenDim: 64, NumLayers: 4, NumHeads: 8, NumKVHeads: 4, HeadDim: 8,
		VocabSize: 128, IntermediateDim: 128, IntermediateSizeMLP: 192, RMSNormEps: 1e-5,
		NumLocalExperts: 4, NumExpertsPerTok: 1, MoeLayers: []int{1, 3}, NoRopeLayers: []int{1, 1, 0, 1},
		UseQKNorm: true, AttnTemperatureTuning: true, FloorScale: 4, AttnScaleL4: 0.1,
		RopeParameters: json.RawMessage(`{"rope_theta":10000.0,"rope_type":"default"}`),
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if schema != &llama4TensorSchema || arch.llama4 == nil {
		t.Fatalf("schema/llama4 mismatch")
	}
	lp := arch.llama4
	if got := []bool{lp.useRope[0], lp.useRope[1], lp.useRope[2], lp.useRope[3]}; got[2] != false || !got[0] || !got[1] || !got[3] {
		t.Errorf("useRope = %v, want layer 2 NoPE", got)
	}
	if !lp.isMoE[1] || !lp.isMoE[3] || lp.isMoE[0] || lp.isMoE[2] {
		t.Errorf("isMoE = %v, want layers 1,3", lp.isMoE)
	}
	if !lp.useQKNorm || !lp.attnTemp || lp.floorScale != 4 || lp.attnScale != 0.1 {
		t.Errorf("iRoPE params wrong: %+v", lp)
	}
	if !arch.MoE.RouterSigmoid || arch.MoE.NormTopKProb || !arch.MoE.SharedUngated || arch.MoE.RoutedScale != 1 {
		t.Errorf("MoE routing wrong: %+v", arch.MoE)
	}
	if arch.IntermediateDim != 192 || arch.MoE.IntermediateDim != 128 || arch.MoE.SharedIntermediateDim != 128 {
		t.Errorf("width split wrong: dense=%d expert=%d shared=%d", arch.IntermediateDim, arch.MoE.IntermediateDim, arch.MoE.SharedIntermediateDim)
	}
}

// TestLlama4_textParity loads the tiny Llama 4 checkpoint and gates the full iRoPE forward
// (last-logit cosine + greedy continuation, both m.forward and Generate) against the HF
// golden. Exercises RoPE + NoPE layers, L2 QK-norm, attn-temperature, the dense/MoE
// interleave, top-1 sigmoid routing + shared expert, and the transposed fused experts.
// Fixture: scripts/pin_llama4_tiny.py.
func TestLlama4_textParity(t *testing.T) {
	const golden = "../testdata/llama4_tiny_text_golden.json"
	const ckpt = "../testdata/llama4-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_llama4_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_llama4_tiny.py", ckpt)
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
	if m.w.arch.Name != "llama4_text" || m.w.arch.llama4 == nil {
		t.Fatalf("arch = %q (llama4=%v), want llama4_text", m.w.arch.Name, m.w.arch.llama4 != nil)
	}

	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("llama4 text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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

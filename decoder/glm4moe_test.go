package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestGlm4Moe_descriptor checks the glm4_moe adapter resolves the GLM-specific
// seams from config alone (no checkpoint): DeepSeek-style routing flags, the dense
// prefix, partial RoPE, QK-norm, and the ungated shared expert.
func TestGlm4Moe_descriptor(t *testing.T) {
	cfg := &Config{
		ModelType: "glm4_moe", HiddenDim: 64, NumLayers: 3, NumHeads: 4,
		NumKVHeads: 2, HeadDim: 16, VocabSize: 256, IntermediateDim: 128,
		RMSNormEps: 1e-6, NRoutedExperts: 8, NumExpertsPerTok: 2, NSharedExperts: 1,
		MoeIntermediateSize: 32, FirstKDenseReplace: 1, RoutedScalingFactor: 1.0,
		UseQKNorm: true, PartialRotaryFactor: 0.5,
		RopeParameters: json.RawMessage(`{"rope_type":"default","rope_theta":10000.0,"partial_rotary_factor":0.5}`),
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if schema != &glm4moeTensorSchema {
		t.Fatalf("schema = %p, want glm4moeTensorSchema", schema)
	}
	if arch.MoE == nil {
		t.Fatal("MoE descriptor nil")
	}
	if !arch.MoE.RouterSigmoid || !arch.MoE.SharedUngated {
		t.Errorf("RouterSigmoid=%v SharedUngated=%v, want both true", arch.MoE.RouterSigmoid, arch.MoE.SharedUngated)
	}
	if arch.MoE.NumExperts != 8 || arch.MoE.TopK != 2 {
		t.Errorf("experts=%d topk=%d, want 8/2", arch.MoE.NumExperts, arch.MoE.TopK)
	}
	if want := 1 * 32; arch.MoE.SharedIntermediateDim != want {
		t.Errorf("SharedIntermediateDim=%d, want %d (n_shared × moe_inter)", arch.MoE.SharedIntermediateDim, want)
	}
	if arch.FirstKDense != 1 {
		t.Errorf("FirstKDense=%d, want 1", arch.FirstKDense)
	}
	if !arch.QKNorm || arch.QKVBias {
		t.Errorf("QKNorm=%v QKVBias=%v, want true/false", arch.QKNorm, arch.QKVBias)
	}
	if want := int(0.5 * 16); arch.RotaryDim != want {
		t.Errorf("RotaryDim=%d, want %d (partial_rotary · head_dim)", arch.RotaryDim, want)
	}
	if arch.RoPEGlobalBase != 10000 {
		t.Errorf("RoPEGlobalBase=%v, want 10000 (from rope_parameters)", arch.RoPEGlobalBase)
	}
}

// TestGlm4Moe_biasParity gates the REAL GLM-4.5-Air flag combination the default
// tiny doesn't: attention_bias=True (q/k/v bias) + use_qk_norm=False. If this is
// cosine 1.0, the bias / no-QK-norm forward is correct and any real-model roughness
// is quantization, not a loader bug. Fixture: scripts/pin_glm_tiny_bias.py.
func TestGlm4Moe_biasParity(t *testing.T) {
	const golden = "../testdata/glm_tiny_bias_text_golden.json"
	const ckpt = "../testdata/glm-tiny-bias"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_glm_tiny_bias.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_glm_tiny_bias.py", ckpt)
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
	if !m.w.arch.QKVBias || m.w.arch.QKNorm {
		t.Fatalf("arch flags QKVBias=%v QKNorm=%v, want true/false", m.w.arch.QKVBias, m.w.arch.QKNorm)
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
	t.Logf("glm4_moe bias parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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
	// tiny-golden: goinfer vs the HF f32 forward of the seeded glm-tiny — exact numeric oracle for
	// the sigmoid-routing + shared-expert + first_k_dense forward (real GLM-4.5-Air is 106B, bf16
	// ~212 GB > 62 GB RAM, so a real-model forward is infeasible on this box).
	emitParityRow(t, "glm4_moe", "tiny-golden", "HF f32 (glm-tiny seeded fixture)", 100.0, float64(cos), float64(cos))
}

// TestGlm4Moe_textParity loads the tiny-random Glm4MoeForCausalLM checkpoint and
// gates the full forward pass — last-logit cosine + greedy continuation — against
// the HF golden. Exercises every GLM seam at once: dense layer 0, sigmoid routing
// with e_score_correction_bias, the ungated shared expert, partial RoPE and QK-norm.
func TestGlm4Moe_textParity(t *testing.T) {
	const golden = "../testdata/glm_tiny_text_golden.json"
	const ckpt = "../testdata/glm-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_glm_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_glm_tiny.py", ckpt)
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
	t.Logf("glm4_moe text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999", cos)
	}

	// Greedy continuation must match the HF golden token-for-token.
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
	// tiny-golden: goinfer vs the HF f32 forward of the seeded glm-tiny — exact numeric oracle for
	// the sigmoid-routing + shared-expert + first_k_dense forward (real GLM-4.5-Air is 106B, bf16
	// ~212 GB > 62 GB RAM, so a real-model forward is infeasible on this box).
	emitParityRow(t, "glm4_moe", "tiny-golden", "HF f32 (glm-tiny seeded fixture)", 100.0, float64(cos), float64(cos))
}

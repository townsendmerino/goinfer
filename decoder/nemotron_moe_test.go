package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestNemotron3NanoMoE_descriptor checks the nemotron_h adapter resolves a FOURTH
// block kind (moe) from the real released spelling — hybrid_override_pattern's "E"
// character, not layers_block_type's "moe" string (every released NemotronH
// checkpoint carries the former, per TestNemotron_descriptor's own precedent) — and
// that arch.MoE is populated from the real Nemotron 3 Nano config field names,
// including the one field verified NOT to alias an existing one:
// moe_shared_expert_intermediate_size is its own field, not
// NSharedExperts*MoeIntermediateSize (DeepSeek's derivation), because the real
// checkpoint ships n_shared_experts=1, moe_intermediate_size=1856 but
// moe_shared_expert_intermediate_size=3712 — 2x, not 1x.
func TestNemotron3NanoMoE_descriptor(t *testing.T) {
	cfg := &Config{
		ModelType: "nemotron_h", HiddenDim: 64, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		VocabSize: 256, IntermediateDim: 128, LayerNormEpsilon: 1e-5,
		MambaNumHeads: 8, MambaHeadDim: 8, SSMStateSize: 16, NGroups: 1, ConvKernel: 4,
		HybridOverridePattern: "M*E",
		NRoutedExperts:        4, NumExpertsPerTok: 2, MoeIntermediateSize: 8,
		NSharedExperts: 1, MoeSharedExpertIntermediateSize: 16, // 2x MoeIntermediateSize, deliberately NOT 1x
		RoutedScalingFactor: 2.5, NGroup: 1, TopkGroup: 1,
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if schema != &nemotronTensorSchema || arch.nemotron == nil {
		t.Fatalf("schema/nemotron mismatch")
	}
	want := []uint8{nemoMamba, nemoAttn, nemoMoE}
	for i, k := range want {
		if arch.nemotron.blockKind[i] != k {
			t.Errorf("blockKind[%d]=%d, want %d", i, arch.nemotron.blockKind[i], k)
		}
	}
	if arch.MoE == nil {
		t.Fatal("arch.MoE is nil, want populated (pattern has an moe block)")
	}
	m := arch.MoE
	switch {
	case m.NumExperts != 4:
		t.Errorf("NumExperts=%d, want 4", m.NumExperts)
	case m.TopK != 2:
		t.Errorf("TopK=%d, want 2", m.TopK)
	case m.IntermediateDim != 8:
		t.Errorf("IntermediateDim=%d, want 8", m.IntermediateDim)
	case m.SharedIntermediateDim != 16:
		t.Errorf("SharedIntermediateDim=%d, want 16 (moe_shared_expert_intermediate_size, NOT NSharedExperts*MoeIntermediateSize=8)", m.SharedIntermediateDim)
	case !m.RouterSigmoid:
		t.Error("RouterSigmoid=false, want true (unconditional for nemotron_h, no scoring_func key exists)")
	case m.RoutedScale != 2.5:
		t.Errorf("RoutedScale=%v, want 2.5", m.RoutedScale)
	case !m.SharedUngated:
		t.Error("SharedUngated=false, want true (plain additive shared expert)")
	case m.NGroup != 1 || m.TopkGroup != 1:
		t.Errorf("NGroup=%d TopkGroup=%d, want 1/1", m.NGroup, m.TopkGroup)
	}
	// Plain (non-MoE) Nemotron-H must be completely unaffected: no "E" in the
	// pattern ⇒ arch.MoE stays nil, exactly as before this change.
	plainCfg := &Config{
		ModelType: "nemotron_h", HiddenDim: 64, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		VocabSize: 256, IntermediateDim: 128, LayerNormEpsilon: 1e-5,
		MambaNumHeads: 8, MambaHeadDim: 8, SSMStateSize: 16, NGroups: 1, ConvKernel: 4,
		HybridOverridePattern: "M*-",
	}
	plainArch, _, err := resolveArchitecture(plainCfg)
	if err != nil {
		t.Fatalf("resolveArchitecture (plain): %v", err)
	}
	if plainArch.MoE != nil {
		t.Errorf("plain Nemotron-H got arch.MoE=%+v, want nil", plainArch.MoE)
	}
}

// TestNemotron3NanoMoE_forward hand-computes nemotronMoE's expected output —
// exercising the actual dispatch path (sigmoid routing + non-gated relu² routed
// experts + an ungated additive shared expert), not just the config wiring. TopK=2
// over 2 experts (both selected, so the weighted-sum combine is genuinely
// exercised, not trivially collapsed to weight=1 the way a TopK=1-of-N case would
// be) with NormTopKProb=true renormalizing the sigmoid scores to sum 1.
//
//	router logits = Router·n = [2, 1]  (n = [1,1])
//	w0 = sigmoid(2)/(sigmoid(2)+sigmoid(1)),  w1 = sigmoid(1)/(sigmoid(2)+sigmoid(1))
//	expert0: up=[1,1] -> relu2=[1,1] -> down=[2,2]
//	expert1: up=[2,2] -> relu2=[4,4] -> down=[4,4]
//	shared:  up=[2,2] -> relu2=[4,4] -> down=[4,4]  (added ungated, no weight)
//	want = [2*w0+4*w1+4, 2*w0+4*w1+4]
func TestNemotron3NanoMoE_forward(t *testing.T) {
	hidden, inter := 2, 2
	arch := &Architecture{
		HiddenDim: hidden,
		MoE: &MoEConfig{
			NumExperts: 2, TopK: 2, NormTopKProb: true, IntermediateDim: inter,
			SharedIntermediateDim: inter, RouterSigmoid: true, RoutedScale: 1.0, SharedUngated: true,
		},
	}
	lw := &LayerWeights{
		Router:     linalg.WrapF32([]float32{2, 0, 0, 1}, 2, hidden), // logits = [2n0, n1]
		RouterBias: []float32{0, 0},
		Experts: []expertWeights{
			{
				Up:   linalg.WrapF32([]float32{1, 0, 0, 1}, inter, hidden), // identity
				Down: linalg.WrapF32([]float32{1, 1, 1, 1}, hidden, inter), // row-sum
			},
			{
				Up:   linalg.WrapF32([]float32{2, 0, 0, 2}, inter, hidden), // 2x identity
				Down: linalg.WrapF32([]float32{1, 0, 0, 1}, hidden, inter), // identity
			},
		},
		SharedExpert: expertWeights{
			Up:   linalg.WrapF32([]float32{1, 1, 1, 1}, inter, hidden), // row-sum
			Down: linalg.WrapF32([]float32{1, 0, 0, 1}, hidden, inter), // identity
		},
	}
	m := &Model{be: &cpuBackend{}}
	n := []float32{1, 1}
	got := m.nemotronMoE(n, lw, arch, hidden)

	s0 := 1 / (1 + math.Exp(-2))
	s1 := 1 / (1 + math.Exp(-1))
	w0, w1 := s0/(s0+s1), s1/(s0+s1)
	want := float32(2*w0+4*w1) + 4

	for i, v := range got {
		if d := math.Abs(float64(v - want)); d > 1e-5 {
			t.Errorf("got[%d]=%v, want %v (diff %v)", i, v, want, d)
		}
	}
}

// TestNemotron3NanoMoE_textParity is the real T1: loads a tiny-random NemotronH-MoE
// checkpoint (scripts/pin_nemotron3nano_tiny.py, transformers' own NemotronHConfig/
// NemotronHForCausalLM with n_routed_experts/moe_intermediate_size/etc set) and gates
// the full forward (last-logit cosine + greedy continuation, both m.forward and
// Generate) against the real HF oracle — not the hand-computed unit tests above,
// which check the Go implementation is internally consistent with itself; this checks
// it against transformers' actual forward pass. Exercises mamba + attention + moe in
// one checkpoint (pattern mamba/moe/mamba/attention/mamba/moe — the real Nemotron 3
// Nano pattern has no plain dense-mlp layer at all, so neither does this fixture).
func TestNemotron3NanoMoE_textParity(t *testing.T) {
	const golden = "../testdata/nemotron3nano_tiny_text_golden.json"
	const ckpt = "../testdata/nemotron3nano-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_nemotron3nano_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_nemotron3nano_tiny.py", ckpt)
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
	t.Logf("nemotron3nano-moe text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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

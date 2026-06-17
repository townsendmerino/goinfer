package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestDeepseek_descriptor checks the deepseek_v3 adapter resolves the MLA + DeepSeekMoE
// seams from config alone: the split MLA head dims, the q-LoRA bottleneck, the latent
// width, sigmoid group-limited routing (n_group/topk_group + routed scaling), the ungated
// shared expert, and the first_k_dense_replace dense prefix.
func TestDeepseek_descriptor(t *testing.T) {
	cfg := &Config{
		ModelType: "deepseek_v3", HiddenDim: 64, NumLayers: 4, NumHeads: 4, NumKVHeads: 4,
		VocabSize: 128, IntermediateDim: 128, MoeIntermediateSize: 32, RMSNormEps: 1e-6,
		QLoRARank: 24, KVLoRARank: 16, QKNopeHeadDim: 16, QKRopeHeadDim: 8, VHeadDim: 16,
		NRoutedExperts: 8, NSharedExperts: 1, NumExpertsPerTok: 2, NGroup: 2, TopkGroup: 1,
		FirstKDenseReplace: 1, RoutedScalingFactor: 2.5,
		RopeParameters: json.RawMessage(`{"rope_theta":10000.0,"rope_type":"default"}`),
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if schema != &deepseekTensorSchema || arch.mla == nil {
		t.Fatalf("schema/mla mismatch")
	}
	p := arch.mla
	if p.QLoRARank != 24 || p.KVLoRARank != 16 || p.QKNopeHeadDim != 16 || p.QKRopeHeadDim != 8 || p.VHeadDim != 16 {
		t.Errorf("MLA dims wrong: %+v", p)
	}
	if p.qkHeadDim() != 24 || arch.HeadDim != 24 {
		t.Errorf("qk_head_dim=%d arch.HeadDim=%d, want 24", p.qkHeadDim(), arch.HeadDim)
	}
	if arch.RotaryDim != 8 {
		t.Errorf("RotaryDim=%d, want 8 (rope rotates only the rope-carrying dims)", arch.RotaryDim)
	}
	if !p.ropeInterleave {
		t.Errorf("rope_interleave should default true (V3)")
	}
	if !arch.MoE.RouterSigmoid || arch.MoE.NGroup != 2 || arch.MoE.TopkGroup != 1 || !arch.MoE.SharedUngated {
		t.Errorf("MoE routing wrong: %+v", arch.MoE)
	}
	if arch.MoE.RoutedScale != 2.5 || arch.FirstKDense != 1 {
		t.Errorf("RoutedScale=%v FirstKDense=%d, want 2.5/1", arch.MoE.RoutedScale, arch.FirstKDense)
	}
}

// TestDeepseek_v2lite_descriptor checks the deepseek_v2 routing defaults differ from V3:
// softmax scoring (not sigmoid), no q-LoRA (V2-Lite drops it), n_group≤1 ⇒ plain top-k.
func TestDeepseek_v2lite_descriptor(t *testing.T) {
	cfg := &Config{
		ModelType: "deepseek_v2", HiddenDim: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 4,
		VocabSize: 128, IntermediateDim: 128, MoeIntermediateSize: 32, RMSNormEps: 1e-6,
		QLoRARank: 0, KVLoRARank: 16, QKNopeHeadDim: 16, QKRopeHeadDim: 8, VHeadDim: 16,
		NRoutedExperts: 8, NSharedExperts: 2, NumExpertsPerTok: 2, FirstKDenseReplace: 1,
		RoPEGlobalBase: 10000.0,
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if arch.mla.QLoRARank != 0 {
		t.Errorf("V2-Lite should have no q-LoRA, got %d", arch.mla.QLoRARank)
	}
	if arch.MoE.RouterSigmoid {
		t.Errorf("V2 should use softmax scoring, not sigmoid")
	}
	if arch.MoE.NGroup > 1 {
		t.Errorf("V2-Lite has no group routing, got NGroup=%d", arch.MoE.NGroup)
	}
}

// TestDeepseek_textParity loads the tiny-random DeepSeek-V3 checkpoint and gates the full
// MLA + DeepSeekMoE forward (last-logit cosine + greedy continuation, both m.forward and
// Generate) against the HF golden. Exercises the q-LoRA bottleneck, the latent KV cache,
// decoupled interleaved RoPE, sigmoid group-limited routing, and the dense prefix.
// Fixture: scripts/pin_deepseek_tiny.py.
func TestDeepseek_textParity(t *testing.T) {
	const golden = "../testdata/deepseek_tiny_text_golden.json"
	const ckpt = "../testdata/deepseek-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_deepseek_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_deepseek_tiny.py", ckpt)
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
	t.Logf("deepseek text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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

// TestMLAAbsorb_parity gates the MLA "absorb" perf path (the production default) against
// the naive reconstruct-per-key reference: across a greedy continuation they must give
// the SAME argmax at every step and a last-logit cosine ~1.0. The two are algebraically
// identical (absorb folds kv_b_proj's W_UK into q and W_UV into the output, never
// reconstructing per-head K/V), differing only by f32 reordering. The real-model gates
// (V2-Lite / Moonlight) confirm absorb still matches HF on real weights.
func TestMLAAbsorb_parity(t *testing.T) {
	const golden = "../testdata/deepseek_tiny_text_golden.json"
	const ckpt = "../testdata/deepseek-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_deepseek_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_deepseek_tiny.py", ckpt)
	}
	var g struct {
		PromptIDs []int `json:"prompt_ids"`
		NNew      int   `json:"n_new"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	// run does prompt forward + g.NNew greedy continuation steps, returning the argmax
	// at each step and the final last-token logits — under absorb or naive MLA.
	run := func(naive bool) ([]int, []float32) {
		m, err := Load(ckpt, Options{})
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		defer m.Close()
		mlaForceNaive = naive
		defer func() { mlaForceNaive = false }()
		cache := m.NewCache(len(g.PromptIDs) + g.NNew + 1)
		var logits []float32
		for _, id := range g.PromptIDs {
			if logits, err = m.forward(id, cache); err != nil {
				t.Fatalf("forward: %v", err)
			}
		}
		seq := make([]int, 0, g.NNew)
		for range g.NNew {
			id := argmax(logits)
			seq = append(seq, id)
			if logits, err = m.forward(id, cache); err != nil {
				t.Fatalf("continuation forward: %v", err)
			}
		}
		return seq, logits
	}

	absSeq, absLogits := run(false)
	naiveSeq, naiveLogits := run(true)

	cos := logitCosine(absLogits, naiveLogits)
	t.Logf("MLA absorb vs naive: argmax seq abs=%v naive=%v | last-logit cosine=%.6f", absSeq, naiveSeq, cos)
	for i := range naiveSeq {
		if absSeq[i] != naiveSeq[i] {
			t.Fatalf("absorb/naive argmax diverged at step %d: abs=%d naive=%d", i, absSeq[i], naiveSeq[i])
		}
	}
	if cos < 0.9999 {
		t.Errorf("absorb vs naive last-logit cosine %.6f < 0.9999", cos)
	}
}

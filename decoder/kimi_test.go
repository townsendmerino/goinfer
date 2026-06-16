package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestKimi_descriptor checks the kimi_k2 model_type resolves to the DeepSeek MLA descriptor
// with V3-style routing — sigmoid + e_score_correction_bias, no group limiting (n_group=1),
// q-LoRA, the asymmetric MLA head dims — driven entirely by config (Kimi rides the
// deepseek_v3 path; the deltas vs V3 are scalars: more experts, fewer heads).
func TestKimi_descriptor(t *testing.T) {
	cfg := &Config{
		ModelType: "kimi_k2", HiddenDim: 64, NumLayers: 4, NumHeads: 8, NumKVHeads: 8,
		VocabSize: 128, IntermediateDim: 128, MoeIntermediateSize: 32, RMSNormEps: 1e-6,
		QLoRARank: 24, KVLoRARank: 16, QKNopeHeadDim: 16, QKRopeHeadDim: 8, VHeadDim: 16,
		NRoutedExperts: 16, NSharedExperts: 1, NumExpertsPerTok: 4, NGroup: 1, TopkGroup: 1,
		FirstKDenseReplace: 1, RoutedScalingFactor: 2.827, ScoringFunc: "sigmoid",
		RopeParameters: json.RawMessage(`{"rope_theta":50000.0,"rope_type":"default"}`),
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if schema != &deepseekTensorSchema || arch.mla == nil {
		t.Fatalf("kimi_k2 should resolve to the deepseek MLA descriptor")
	}
	if !arch.MoE.RouterSigmoid || arch.MoE.NGroup > 1 || !arch.MoE.SharedUngated {
		t.Errorf("kimi routing wrong: sigmoid=%v nGroup=%d sharedUngated=%v", arch.MoE.RouterSigmoid, arch.MoE.NGroup, arch.MoE.SharedUngated)
	}
	if arch.MoE.RoutedScale != 2.827 || arch.MoE.NumExperts != 16 || arch.MoE.TopK != 4 {
		t.Errorf("kimi MoE scalars wrong: scale=%v experts=%d topk=%d", arch.MoE.RoutedScale, arch.MoE.NumExperts, arch.MoE.TopK)
	}
	if arch.mla.QLoRARank != 24 || arch.mla.qkHeadDim() != 24 {
		t.Errorf("kimi MLA dims wrong: qLoRA=%d qkHead=%d", arch.mla.QLoRARank, arch.mla.qkHeadDim())
	}
}

// TestKimi_routingDefault checks the sigmoid default fires for kimi_k2 even if a variant's
// config omits scoring_func (the silent-softmax-fallback gotcha — Kimi is always sigmoid).
func TestKimi_routingDefault(t *testing.T) {
	cfg := &Config{
		ModelType: "kimi_k2", HiddenDim: 64, NumLayers: 4, NumHeads: 8, NumKVHeads: 8,
		VocabSize: 128, IntermediateDim: 128, MoeIntermediateSize: 32, RMSNormEps: 1e-6,
		QLoRARank: 24, KVLoRARank: 16, QKNopeHeadDim: 16, QKRopeHeadDim: 8, VHeadDim: 16,
		NRoutedExperts: 16, NSharedExperts: 1, NumExpertsPerTok: 4, FirstKDenseReplace: 1,
		RoPEGlobalBase: 50000, // no ScoringFunc set
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture: %v", err)
	}
	if !arch.MoE.RouterSigmoid {
		t.Errorf("kimi_k2 must default to sigmoid routing when scoring_func is absent")
	}
}

// TestKimi_textParity loads the tiny Kimi-K2-shaped checkpoint (model_type kimi_k2) and
// gates the full MLA + sigmoid-routed DeepSeekMoE forward (last-logit cosine + greedy
// continuation, both m.forward and Generate) against the HF golden. Confirms the kimi_k2
// alias + the config-scalar deltas (more experts, higher top-k, q-LoRA) flow through the
// shipped deepseek path. Fixture: scripts/pin_kimi_tiny.py.
func TestKimi_textParity(t *testing.T) {
	const golden = "../testdata/kimi_tiny_text_golden.json"
	const ckpt = "../testdata/kimi-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_kimi_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_kimi_tiny.py", ckpt)
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
	if m.w.arch.Name != "kimi_k2" || m.w.arch.mla == nil {
		t.Fatalf("arch = %q (mla=%v), want kimi_k2", m.w.arch.Name, m.w.arch.mla != nil)
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
	t.Logf("kimi text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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

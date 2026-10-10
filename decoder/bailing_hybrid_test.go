package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// Bailing Hybrid parity (G5, docs/completed/task-families-2026-09.md; inclusionAI Ling 3.0, model_type
// "bailing_hybrid"): DeepSeek-style Multi-head Latent Attention alternating with Kimi Delta Attention (KDA),
// every layer_group_size-th layer being MLA, over a DeepSeekMoE FFN. MLA and the MoE router compose the existing
// deepseekArchitecture primitives (two parameterized naming departures: both mixers are self.attention, and
// MLA's output projection is self.dense; plus an optional sigmoid output gate). KDA is the new primitive: a
// delta-rule recurrence like Gated DeltaNet with a PER-CHANNEL decay (decoder/kda_rehearsal.go).
//
// Regenerate (hand-assembled tiny checkpoint + golden); scripts/pin_bailing_hybrid_tiny.py's docstring says why the
// real BailingMoeV3ForCausalLM cannot be instantiated on the Mac (its modeling file imports fla.ops.kda, which
// needs Triton):
//
//	~/.venv-nemotron3/bin/python scripts/pin_bailing_hybrid_tiny.py
const (
	bailingHybridModelDir        = "../testdata/bailing_hybrid-tiny"
	bailingHybridForwardGolden   = "../testdata/bailing_hybrid_forward_golden.json"
	bailingHybridForwardFullPath = "../testdata/bailing_hybrid_forward_full.json"
)

func TestBailingHybrid_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs BailingHybrid-tiny")
	}
	raw, err := os.ReadFile(bailingHybridForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no BailingHybrid golden at %s — regenerate with scripts/pin_bailing_hybrid_tiny.py", bailingHybridForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(bailingHybridModelDir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no BailingHybrid checkpoint at %s — regenerate with scripts/pin_bailing_hybrid_tiny.py", bailingHybridModelDir)
	}

	m, err := Load(bailingHybridModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "bailing_hybrid" {
		t.Fatalf("resolved arch %q, want bailing_hybrid", m.w.arch.Name)
	}
	if m.w.arch.kda == nil || m.w.arch.mla == nil {
		t.Fatalf("arch.kda/mla = %v/%v, want both set", m.w.arch.kda, m.w.arch.mla)
	}
	if m.w.arch.NormPlacement != NormPre2 {
		t.Errorf("NormPlacement = %v, want NormPre2 (uniform for both mixer kinds)", m.w.arch.NormPlacement)
	}

	// Layer 3 is this fixture's one MLA layer (layer_group_size=4, matching the release's own
	// 3:1 ratio): must have loaded MLA weights, no KDA state.
	if m.w.Layers[3].mla == nil {
		t.Fatal("layer 3 (MLA) has no MLA weights loaded")
	}
	if m.w.Layers[3].kda != nil {
		t.Error("layer 3 (MLA) loaded KDA weights — should be nil")
	}
	// Layer 0 is a KDA layer: must have loaded KDA weights, no MLA weights.
	if m.w.Layers[0].kda == nil {
		t.Fatal("layer 0 (KDA) has no KDA weights loaded")
	}
	if m.w.Layers[0].mla != nil {
		t.Error("layer 0 (KDA) loaded MLA weights — should be nil")
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

	cos := fullCosine(t, logits, bailingHybridForwardFullPath)
	t.Logf("bailing_hybrid: argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v",
		argmax(logits), g.Argmax, maxSampleΔ, cos)
	emitParityRow(t, "bailing_hybrid", "tiny-golden", "hand-assembled f32 reference (bailing_hybrid-tiny fixture, MLA+KDA composition, KDA's per-channel-decay recurrence via fla-org's own naive reference)", 100.0, cos, cos)
}

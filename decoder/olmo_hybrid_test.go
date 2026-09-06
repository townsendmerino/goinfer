package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// G2 (docs/task-families-2026-09.md, batch 2) Olmo Hybrid parity (allenai/Olmo-Hybrid-7B,
// model_type "olmo_hybrid"): qwen3_5's Gated DeltaNet (3-of-4 layers) + olmo3's own full-attention
// shape (whole-vector QK-norm, 1-of-4 layers) — but NOT a straight composition. The real reason G2
// paused for a design decision: norm placement differs BY LAYER KIND within one model
// (full-attention layers use olmo3's NormPostOnly, DeltaNet layers use plain NormPre2), which
// Architecture.NormPlacementLinear exists to express, keyed on the same layerIsLinear hook that
// already selects the mixer.
//
// Every other departure from a straight qwen3_5-DeltaNet + olmo3-attention composition is a
// parameterization of shared code, not new math: linear_allow_neg_eigval doubles the write-gate
// beta after the sigmoid; q_proj/k_proj/v_proj (and the depthwise conv, q_conv1d/k_conv1d/
// v_conv1d) are separate tensors rather than qwen3_5's pre-concatenated ones; the output
// gated-RMSNorm is named o_norm/o_proj (qwen3_5: norm/out_proj) with a HARDCODED 1e-5 epsilon
// independent of the model's own rms_norm_eps (1e-6 here); rope_parameters is {"rope_theta": null}
// on the release, so there is no RoPE anywhere, on any layer.
//
// VERIFIED AGAINST A REAL Olmo-Hybrid-7B CHECKPOINT (HTTP Range on its safetensors header), not
// just modeling_olmo_hybrid.py's source — the source alone, and even a local save_pretrained
// round-trip through this transformers version's own conversion_mapping.py, both produced tensor
// names/splits that do NOT match the real release; see scripts/pin_olmo_hybrid_tiny.py's own
// docstring for the full account.
//
// Regenerate (seeded tiny OlmoHybridForCausalLM checkpoint + golden, both reproducible):
//
//	~/.venv-nemotron3/bin/python scripts/pin_olmo_hybrid_tiny.py
const (
	olmoHybridModelDir        = "../testdata/olmo_hybrid-tiny"
	olmoHybridForwardGolden   = "../testdata/olmo_hybrid_forward_golden.json"
	olmoHybridForwardFullPath = "../testdata/olmo_hybrid_forward_full.json"
)

func TestOlmoHybrid_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs OlmoHybrid-tiny")
	}
	raw, err := os.ReadFile(olmoHybridForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no OlmoHybrid golden at %s — regenerate with scripts/pin_olmo_hybrid_tiny.py", olmoHybridForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(olmoHybridModelDir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no OlmoHybrid checkpoint at %s — regenerate with scripts/pin_olmo_hybrid_tiny.py", olmoHybridModelDir)
	}

	m, err := Load(olmoHybridModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "olmo_hybrid" {
		t.Fatalf("resolved arch %q, want olmo_hybrid", m.w.arch.Name)
	}
	if m.w.arch.qwen35 == nil {
		t.Fatalf("arch.qwen35 is nil, want the DeltaNet geometry set")
	}
	if !m.w.arch.qwen35.NegEigval {
		t.Error("qwen35.NegEigval = false, want true (linear_allow_neg_eigval on the fixture)")
	}
	if m.w.arch.NormPlacement != NormPostOnly {
		t.Errorf("NormPlacement = %v, want NormPostOnly (full-attention layers)", m.w.arch.NormPlacement)
	}
	if m.w.arch.NormPlacementLinear == nil || *m.w.arch.NormPlacementLinear != NormPre2 {
		t.Errorf("NormPlacementLinear = %v, want &NormPre2 (DeltaNet layers)", m.w.arch.NormPlacementLinear)
	}
	if !m.w.arch.QKNorm || !m.w.arch.QKNormWhole {
		t.Errorf("QKNorm/QKNormWhole = %v/%v, want true/true", m.w.arch.QKNorm, m.w.arch.QKNormWhole)
	}

	// Layer 3 is the fixture's one full-attention layer (layer_types has full_attention every
	// 4th, matching the release's own ratio): whole-vector QK-norm weight is [num_heads*head_dim]
	// wide, and it must have loaded a delta-net-free attention (QProj set, no delta state).
	wantQNormLen := m.w.arch.NumHeads * m.w.arch.HeadDim
	if got := len(m.w.Layers[3].QNorm); got != wantQNormLen {
		t.Errorf("layer 3 QNorm length = %d, want %d (num_heads*head_dim)", got, wantQNormLen)
	}
	if m.w.Layers[3].delta != nil {
		t.Error("layer 3 (full-attention) loaded a DeltaNet state — should be nil")
	}
	// Layer 0 is a DeltaNet layer: must have loaded delta state, no plain QProj.
	if m.w.Layers[0].delta == nil {
		t.Fatal("layer 0 (linear) has no DeltaNet weights loaded")
	}
	if m.w.Layers[0].QProj.Rows() != 0 {
		t.Error("layer 0 (linear) loaded a plain QProj — should be absent (DeltaNet-only)")
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

	cos := fullCosine(t, logits, olmoHybridForwardFullPath)
	t.Logf("olmo_hybrid: argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v",
		argmax(logits), g.Argmax, maxSampleΔ, cos)
	emitParityRow(t, "olmo_hybrid", "tiny-golden", "HF f32 (olmo_hybrid-tiny seeded fixture, mixed NormPlacement + NegEigval + separate DeltaNet tensors + NoPE)", 100.0, cos, cos)
}

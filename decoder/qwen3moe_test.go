package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// F1 Qwen3-MoE parity (Qwen3-30B-A3B / Qwen3-Coder-30B-A3B-Instruct, model_type
// "qwen3_moe"): qwen3's dense attention (per-head q_norm/k_norm, GQA, no q/k/v
// bias) with the FFN replaced on every layer by a sparse MoE — qwen2_moe's
// router shape but with NO shared expert. Loads a tiny seeded
// Qwen3MoeForCausalLM through the generic forward and matches the HF float32
// oracle, exercising QK-norm composed with routed-MoE-with-no-shared-expert for
// the first time.
//
// Regenerate (seeded tiny Qwen3MoeForCausalLM checkpoint + golden, both reproducible):
//
//	~/.venv-nemotron3/bin/python scripts/pin_qwen3moe_tiny.py
const (
	qwen3MoeModelDir        = "../testdata/qwen3moe-tiny"
	qwen3MoeForwardGolden   = "../testdata/qwen3moe_forward_golden.json"
	qwen3MoeForwardFullPath = "../testdata/qwen3moe_forward_full.json"
)

func TestQwen3Moe_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs Qwen3-MoE-tiny")
	}
	raw, err := os.ReadFile(qwen3MoeForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Qwen3-MoE golden at %s — regenerate with scripts/pin_qwen3moe_tiny.py", qwen3MoeForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(qwen3MoeModelDir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Qwen3-MoE checkpoint at %s — regenerate with scripts/pin_qwen3moe_tiny.py", qwen3MoeModelDir)
	}

	m, err := Load(qwen3MoeModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "qwen3_moe" {
		t.Fatalf("resolved arch %q, want qwen3_moe", m.w.arch.Name)
	}
	if !m.w.arch.QKNorm {
		t.Fatalf("expected QK-norm")
	}
	if m.w.arch.MoE == nil {
		t.Fatalf("expected a MoE config")
	}
	if m.w.arch.MoE.NumExperts != 8 || m.w.arch.MoE.TopK != 2 {
		t.Errorf("MoE = %dx top-%d, want 8x top-2", m.w.arch.MoE.NumExperts, m.w.arch.MoE.TopK)
	}
	if m.w.arch.MoE.SharedIntermediateDim != 0 {
		t.Errorf("SharedIntermediateDim = %d, want 0 (qwen3_moe has no shared expert)", m.w.arch.MoE.SharedIntermediateDim)
	}
	if got := len(m.w.Layers[0].Experts); got != 8 {
		t.Fatalf("layer 0 loaded %d experts, want 8", got)
	}
	if m.w.Layers[0].SharedExpert.Gate.Rows() != 0 {
		t.Fatalf("layer 0 loaded a shared expert, want none")
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

	cos := fullCosine(t, logits, qwen3MoeForwardFullPath)
	t.Logf("qwen3_moe: %dx top-%d | argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v",
		m.w.arch.MoE.NumExperts, m.w.arch.MoE.TopK, argmax(logits), g.Argmax, maxSampleΔ, cos)
	// tiny-golden: goinfer's forward vs the HF f32 forward of the SEEDED qwen3moe-tiny —
	// an exact numeric oracle for the loader + QK-norm + no-shared-expert MoE routing.
	// The real Qwen3-30B-A3B (bf16 ~61GB) is a Linux-box T3, not yet run here.
	emitParityRow(t, "qwen3_moe", "tiny-golden", "HF f32 (qwen3moe-tiny seeded fixture, 8x top-2, no shared expert)", 100.0, cos, cos)
}

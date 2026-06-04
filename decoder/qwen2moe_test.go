package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// Qwen2-MoE (shared-expert MoE) parity. qwen2_moe is qwen2's attention (q/k/v
// bias, no QK-norm) with the FFN replaced on every layer by a sparse router +
// top-k experts PLUS an always-on shared expert gated by sigmoid(shared_gate·h).
// Validated structurally against HF on a tiny random checkpoint
// (katuni4ka/tiny-random-qwen1.5-moe): random weights, so the predicted token is
// meaningless, but aikit's forward must reproduce HF's f32 logits — which
// exercises the routed-expert sum, the shared expert, and the sigmoid gate.
//
//	hf download katuni4ka/tiny-random-qwen1.5-moe --local-dir testdata/tiny-qwen2-moe
//	.venv/bin/python scripts/pin_llama_forward.py testdata/tiny-qwen2-moe qwen2moe
const (
	qwen2moeModelDir        = "../testdata/tiny-qwen2-moe"
	qwen2moeForwardGolden   = "../testdata/qwen2moe_forward_golden.json"
	qwen2moeForwardFullPath = "../testdata/qwen2moe_forward_full.json"
)

func TestQwen2Moe_forwardParity(t *testing.T) {
	raw, err := os.ReadFile(qwen2moeForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no qwen2_moe golden at %s — regenerate with scripts/pin_llama_forward.py", qwen2moeForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(qwen2moeModelDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no qwen2_moe checkpoint at %s", qwen2moeModelDir)
	}

	m, err := Load(qwen2moeModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "qwen2_moe" {
		t.Fatalf("resolved arch %q, want qwen2_moe", m.w.arch.Name)
	}
	if m.w.arch.MoE == nil || m.w.arch.MoE.SharedIntermediateDim == 0 {
		t.Fatalf("shared-expert MoE not configured (MoE=%v)", m.w.arch.MoE)
	}
	// The routed experts, shared expert, and sigmoid gate all loaded.
	l0 := &m.w.Layers[0]
	if len(l0.Experts) != m.w.arch.MoE.NumExperts {
		t.Fatalf("layer 0 has %d experts, want %d", len(l0.Experts), m.w.arch.MoE.NumExperts)
	}
	if l0.SharedExpert.Gate.rows == 0 || l0.SharedGate.rows == 0 {
		t.Fatalf("shared expert / gate not loaded for layer 0")
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
	var maxΔ float64
	for _, kv := range g.Sample {
		id := int(kv[0])
		if d := math.Abs(float64(logits[id]) - kv[1]); d > maxΔ {
			maxΔ = d
		}
		if d := math.Abs(float64(logits[id]) - kv[1]); d > valTol {
			t.Errorf("sample id=%d logit=%.5f want %.5f (Δ%.5f)", id, logits[id], kv[1], d)
		}
	}
	cos := fullCosine(t, logits, qwen2moeForwardFullPath)
	t.Logf("qwen2_moe: argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v", argmax(logits), g.Argmax, maxΔ, cos)
}

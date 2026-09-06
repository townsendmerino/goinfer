package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// G2 (docs/task-families-2026-09.md, batch 2) Olmo 3 parity (allenai/Olmo-3-{7B,32B}, model_type
// "olmo3"): two real departures from every existing family, verified against the real
// modeling_olmo3.py — NormPostOnly (no pre-norm at all; only the sublayer OUTPUT is normalized
// before the residual add) and QKNormWhole (QK-norm over the FULL projected q/k vector, not
// per-head). Also: YaRN scaling applies ONLY to full_attention layers — sliding_attention layers
// get plain unscaled RoPE at the same theta, confirmed against configuration_olmo3.py's
// convert_rope_params_to_dict (`self.rope_parameters["full_attention"].update(rope_scaling)`,
// leaving "sliding_attention" at its plain default) — the SAME local/global RoPE split Mellum
// already implements, so no new mechanism, just the existing ropeScaling/ropeScalingLocal split
// applied without ever setting ropeScalingLocal (nil ⇒ no scaling on sliding layers).
//
// Regenerate (seeded tiny Olmo3ForCausalLM checkpoint + golden, both reproducible):
//
//	~/.venv-nemotron3/bin/python scripts/pin_olmo3_tiny.py
const (
	olmo3ModelDir        = "../testdata/olmo3-tiny"
	olmo3ForwardGolden   = "../testdata/olmo3_forward_golden.json"
	olmo3ForwardFullPath = "../testdata/olmo3_forward_full.json"
)

func TestOlmo3_forwardParity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + runs Olmo3-tiny")
	}
	raw, err := os.ReadFile(olmo3ForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Olmo3 golden at %s — regenerate with scripts/pin_olmo3_tiny.py", olmo3ForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if _, err := os.Stat(olmo3ModelDir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Olmo3 checkpoint at %s — regenerate with scripts/pin_olmo3_tiny.py", olmo3ModelDir)
	}

	m, err := Load(olmo3ModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.Name != "olmo3" {
		t.Fatalf("resolved arch %q, want olmo3", m.w.arch.Name)
	}
	if m.w.arch.NormPlacement != NormPostOnly {
		t.Errorf("NormPlacement = %v, want NormPostOnly", m.w.arch.NormPlacement)
	}
	if !m.w.arch.QKNorm || !m.w.arch.QKNormWhole {
		t.Errorf("QKNorm/QKNormWhole = %v/%v, want true/true", m.w.arch.QKNorm, m.w.arch.QKNormWhole)
	}
	// The whole-vector QK-norm weight is [num_heads*head_dim] wide, not [head_dim] — confirms the
	// tensor was loaded at the right length, not silently truncated/padded.
	wantQNormLen := m.w.arch.NumHeads * m.w.arch.HeadDim
	if got := len(m.w.Layers[0].QNorm); got != wantQNormLen {
		t.Errorf("layer 0 QNorm length = %d, want %d (num_heads*head_dim)", got, wantQNormLen)
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

	cos := fullCosine(t, logits, olmo3ForwardFullPath)
	t.Logf("olmo3: argmax=%d (want %d) | maxSampleΔ=%.5f | cosine=%v",
		argmax(logits), g.Argmax, maxSampleΔ, cos)
	emitParityRow(t, "olmo3", "tiny-golden", "HF f32 (olmo3-tiny seeded fixture, NormPostOnly + QKNormWhole + YaRN-on-full-only)", 100.0, cos, cos)
}

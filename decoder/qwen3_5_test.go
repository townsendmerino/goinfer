package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// TestQwen3_5_textParity is the tiny-oracle gate for Qwen3.8 (model_type qwen3_5) — the DENSE
// member of the Gated-DeltaNet/softmax hybrid family goinfer already runs as qwen3_5_moe and
// qwen3_next.
//
// The fixture keeps the released model's SHAPE CHARACTER rather than its size, because the
// character is what breaks:
//
//   - head_dim (32) is NOT hidden/num_heads. The real model is head_dim 256 at hidden 5120
//     with 24 heads, so nH·hd = 6144 ≠ hidden — a fixture where they coincided would hide
//     precisely the class of bug this family invites.
//   - 3:1 layer_types, so BOTH mixers run: three Gated-DeltaNet layers and one softmax.
//   - GVA with linear_num_value_heads a multiple of linear_num_key_heads.
//   - mrope_section + mrope_interleaved are present in the config, and the text path must
//     still reduce to plain partial RoPE (verified in transformers' apply_interleaved_mrope:
//     text position_ids arrive 2-D and are expanded to three IDENTICAL components, so the
//     interleaved overwrite is a no-op). If that reduction were wrong, this diverges.
func TestQwen3_5_textParity(t *testing.T) {
	const golden = "testdata/qwen3_5_tiny_text_golden.json"
	const ckpt = "testdata/qwen3_5-tiny"

	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_qwen3_5_forward.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_qwen3_5_forward.py", ckpt)
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

	a := m.w.arch
	if a.Name != "qwen3_5" {
		t.Fatalf("arch = %q, want qwen3_5", a.Name)
	}
	if a.MoE != nil {
		t.Fatal("arch.MoE != nil — this is the DENSE variant; a router here means the wrong adapter ran")
	}
	if a.qwen35 == nil {
		t.Fatal("arch.qwen35 is nil — the DeltaNet geometry must be carried, dense or not")
	}
	// Both mixers must be present, or the gate covers half the model.
	lin, full := 0, 0
	for i := range a.NumLayers {
		if a.isLinearLayer(i) {
			lin++
		} else {
			full++
		}
	}
	if lin == 0 || full == 0 {
		t.Fatalf("layer mix = %d linear / %d full — the fixture must exercise BOTH mixers", lin, full)
	}
	// nH·hd != hidden is the family's signature; assert the descriptor kept them independent.
	if a.NumHeads*a.HeadDim == a.HiddenDim {
		t.Fatalf("nH·hd == hidden (%d) — the fixture no longer exercises the independent-head_dim "+
			"shape this family has", a.HiddenDim)
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
	t.Logf("qwen3_5 text parity (%d linear / %d full): argmax got=%d want=%d | logit cosine=%.6f",
		lin, full, gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999", cos)
	}

	// Greedy continuation: the DeltaNet state is recurrent, so a small per-step error shows
	// as drift rather than in the first logit.
	cur := append([]int(nil), g.PromptIDs...)
	for i := range g.NNew {
		c2 := m.NewCache(len(cur) + 1)
		var lg []float32
		for _, id := range cur {
			if lg, err = m.forward(id, c2); err != nil {
				t.Fatalf("forward: %v", err)
			}
		}
		nxt := argmax(lg)
		if i < len(g.ContinuationIDs) && nxt != g.ContinuationIDs[i] {
			t.Errorf("continuation[%d] = %d, want %d — recurrent state drift", i, nxt, g.ContinuationIDs[i])
			break
		}
		cur = append(cur, nxt)
	}
}

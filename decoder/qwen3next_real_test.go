package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestQwen3Next_textParity is the real T1: loads a tiny-random Qwen3-Next
// checkpoint (scripts/pin_qwen3next_tiny.py) and gates the full forward
// (last-logit cosine + greedy continuation, both m.forward and Generate)
// against the real HF oracle — transformers 5.15.0's actual
// Qwen3NextForCausalLM, no trust_remote_code needed (mainline support).
//
// The fixture's config.json is deliberately rewritten (by the pin script,
// automatically) to the REAL release's shape — full_attention_interval and a
// top-level rope_theta, no layer_types, no rope_parameters — so this test
// exercises normalizeQwen3NextLayerTypes and the flat-RoPE path directly, the
// two real deltas from qwen3_5_moe this family exists to test, not just the
// already-populated shape qwen3_5_moe's own fixture already covers.
func TestQwen3Next_textParity(t *testing.T) {
	const golden = "../testdata/qwen3next_tiny_text_golden.json"
	const ckpt = "../testdata/qwen3next-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_qwen3next_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_qwen3next_tiny.py", ckpt)
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
	t.Logf("qwen3next text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
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

package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"reflect"
	"testing"
)

// TestQwen3VL_textParity is qwen25vl_test.go's TestQwen25VL_textParity twin for the P8 Phase 0
// text-only bring-up (docs/multimodal.md): loads the tiny Qwen3-VL checkpoint (scripts/
// pin_qwen3vl_tiny.py) through goinfer's loader + forward and asserts the TEXT-ONLY path matches
// the HF golden — proving qwen3_vlArchitecture (Qwen3's dense attention shape: per-head q/k
// RMSNorm, GQA, no q/k/v bias — NOT Qwen2's shape, which qwen2_5_vlArchitecture aliases) is wired
// correctly for a Qwen3-VL-nested config. No vision tower, no DeepStack, no image path — those are
// explicitly out of scope for this phase (docs/multimodal.md's P8 entry).
func TestQwen3VL_textParity(t *testing.T) {
	const golden = "../testdata/qwen3vl_tiny_text_golden.json"
	const ckpt = "../testdata/qwen3vl-tiny"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_qwen3vl_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_qwen3vl_tiny.py", ckpt)
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

	if m.w.arch.Name != "qwen3_vl" {
		t.Fatalf("arch.Name = %q, want qwen3_vl", m.w.arch.Name)
	}
	if !m.w.arch.MRopeInterleaved {
		t.Error("arch.MRopeInterleaved = false, want true for qwen3_vl")
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
	t.Logf("qwen3-VL text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999", cos)
	}

	// Greedy continuation must match the HF golden token-for-token.
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
}

// TestMropeComponentInterleaved_matchesHF pins mropeComponentInterleaved against ground truth
// computed directly from the REAL HF slicing logic (Qwen3VLTextRotaryEmbedding.
// recomposition_frequencies, transformers/models/qwen3_vl/modeling_qwen3_vl.py) — not against a
// re-derivation of the formula, and not only against TestQwen3VL_textParity's golden above, which
// cannot exercise this at all: every text position there has equal (t,h,w) components (no image
// token in the prompt), so the component lookup is unreachable-in-effect regardless of which
// formula is used. This test is the ONLY thing in Phase 0 that actually proves
// mropeComponentInterleaved is correct, by construction on genuinely divergent section splits.
//
// Ground truth reproduced via (run once, not part of the test):
//
//	python -c "
//	import torch
//	def recomposition_component(half, section):
//	    freq = torch.zeros(3, half, dtype=torch.long)
//	    for dim in range(3): freq[dim, :] = dim
//	    out = freq[0].clone()
//	    for dim, offset in ((1, 1), (2, 2)):
//	        idx = slice(offset, section[dim] * 3, 3)
//	        out[idx] = freq[dim, idx]
//	    return out.tolist()
//	"
func TestMropeComponentInterleaved_matchesHF(t *testing.T) {
	for _, tc := range []struct {
		name    string
		section []int
		want    []int // component per frequency index d, HF ground truth
	}{
		{"4,2,2 (the tiny golden's own section)", []int{4, 2, 2}, []int{0, 1, 2, 0, 1, 2, 0, 0}},
		{"24,20,20 (Qwen3-VL's real default)", []int{24, 20, 20}, []int{
			0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
			0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
			0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0,
		}},
		{"3,3,2 (asymmetric H>W)", []int{3, 3, 2}, []int{0, 1, 2, 0, 1, 2, 0, 1}},
		{"5,1,2 (asymmetric H<W, H section shorter than its own stride)", []int{5, 1, 2}, []int{0, 1, 2, 0, 0, 2, 0, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			half := 0
			for _, n := range tc.section {
				half += n
			}
			got := make([]int, half)
			for d := range half {
				got[d] = mropeComponentInterleaved(d, tc.section)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("mropeComponentInterleaved(section=%v):\n got  %v\n want %v", tc.section, got, tc.want)
			}
		})
	}
}

// TestMropeComponentInterleaved_differsFromChunked is the adversarial half: on a
// non-trivial section split, the interleaved and chunked (mropeComponent) formulas must
// actually disagree somewhere — otherwise a test asserting either one in isolation could
// pass by accident on a formula that silently fell back to the other.
func TestMropeComponentInterleaved_differsFromChunked(t *testing.T) {
	section := []int{4, 2, 2}
	half := 8
	differs := false
	for d := 0; d < half; d++ {
		if mropeComponentInterleaved(d, section) != mropeComponent(d, section) {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("mropeComponentInterleaved and mropeComponent agree on every index for section [4,2,2] — " +
			"this test cannot distinguish the two formulas, fix the fixture")
	}
}

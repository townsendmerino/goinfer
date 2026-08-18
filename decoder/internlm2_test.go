package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// TestInternLM2_textParity gates the grouped wqkv de-interleave — the only new code this
// family needed.
//
// The fixture is built by taking a reference llama's SEPARATE q/k/v and re-packing them into
// InternLM2's fused layout (per KV head: `groups` query rows, then K, then V), while the
// golden is that same llama's own forward. So the two agree only if goinfer's gather inverts
// the interleave exactly. Reading the tensor as phi3-style [Q ‖ K ‖ V] gives correct shapes,
// finite values, and K rows standing in for query heads — which is why the check is a forward
// against a reference rather than a shape assertion.
//
// The geometry is chosen so the interleave cannot degenerate: 8 heads over 2 KV heads means
// groups = 4 and gs = 6, so Q, K and V rows genuinely alternate. With num_key_value_heads = 1
// (or groups = 1) the grouped layout and a plain concat coincide, and this test would pass
// against the wrong reader.
func TestInternLM2_textParity(t *testing.T) {
	const golden = "testdata/internlm2_tiny_text_golden.json"
	const ckpt = "testdata/internlm2-tiny"

	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_internlm2_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_internlm2_tiny.py", ckpt)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
		Groups          int       `json:"groups"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if g.Groups < 2 {
		t.Fatalf("fixture has groups=%d: with groups<2 the grouped layout and a plain concat "+
			"coincide, so this gate would pass against the wrong reader", g.Groups)
	}

	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", ckpt, err)
	}
	defer m.Close()

	if got := m.w.arch.Name; got != "internlm2" {
		t.Fatalf("arch = %q, want internlm2", got)
	}
	// The split must produce the ordinary per-projection shapes the generic forward expects.
	a := m.w.arch
	if got := m.w.Layers[0].QProj.Rows(); got != a.NumHeads*a.HeadDim {
		t.Errorf("q rows = %d, want %d", got, a.NumHeads*a.HeadDim)
	}
	if got := m.w.Layers[0].KProj.Rows(); got != a.NumKVHeads*a.HeadDim {
		t.Errorf("k rows = %d, want %d", got, a.NumKVHeads*a.HeadDim)
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
	t.Logf("internlm2 text parity (groups=%d): argmax got=%d want=%d | logit cosine=%.6f",
		g.Groups, gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999 — a wrong wqkv gather lands far below this, "+
			"not just outside tolerance", cos)
	}

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
			t.Errorf("continuation[%d] = %d, want %d", i, nxt, g.ContinuationIDs[i])
			break
		}
		cur = append(cur, nxt)
	}
}

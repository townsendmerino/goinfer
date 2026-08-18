package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// TestInternLM3_textParity gates the ONE thing that makes internlm3 its own model_type.
//
// The family is a registry ALIAS onto llamaArchitecture — llama-shaped down to the tensor
// names (self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, input_layernorm, embed_tokens /
// lm_head, no biases, no QK-norm) — so its SHAPE is already covered by llama's own oracle and
// a second shape gate would test nothing new.
//
// What llama's oracle does not exercise is `rope_scaling: {"rope_type": "dynamic"}`. goinfer
// accepts that as no scaling at all, on the grounds that HF's DynamicNTKScalingRotaryEmbedding
// rescales ONLY once seq_len exceeds max_position_embeddings — so below that boundary it is
// identity, not an approximation. That is a claim about someone else's code, and claims about
// someone else's code get a reference: the fixture is a real LlamaForCausalLM carrying that
// exact rope_scaling, so HF applies its dynamic path and goinfer must still match.
//
// LIMITATION, stated because the gate cannot see it: beyond max_position_embeddings the two
// diverge. Dynamic NTK needs a per-sequence inv-freq rebuild, which the precomputed
// finalizeRoPE tables cannot express. This test runs entirely in-window by construction.
func TestInternLM3_textParity(t *testing.T) {
	const golden = "testdata/internlm3_tiny_text_golden.json"
	const ckpt = "testdata/internlm3-tiny"

	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/pin_internlm3_tiny.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint at %s — run scripts/pin_internlm3_tiny.py", ckpt)
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
		t.Fatalf("Load(%s): %v — a dynamic rope_scaling must LOAD, not error", ckpt, err)
	}
	defer m.Close()

	// The alias resolves onto llama's descriptor, and the dynamic scaling resolves to none.
	if got := m.w.arch.Name; got != "llama" {
		t.Errorf("arch = %q, want llama (internlm3 is an alias, not its own descriptor)", got)
	}
	if m.w.arch.ropeScaling != nil {
		t.Errorf("ropeScaling = %+v, want nil — dynamic NTK is identity within the trained "+
			"window, so it must resolve to no scaling rather than to some approximation",
			m.w.arch.ropeScaling)
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
	t.Logf("internlm3 text parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999", cos)
	}

	// Greedy continuation: a rope difference shows as drift across positions, not in one logit.
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
			t.Errorf("continuation[%d] = %d, want %d — positional drift is what a wrong rope "+
				"reading looks like", i, nxt, g.ContinuationIDs[i])
			break
		}
		cur = append(cur, nxt)
	}
}

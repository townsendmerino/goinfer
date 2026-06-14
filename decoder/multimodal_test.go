package decoder

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// Multimodal P0 — the embed-by-vector forward seam. These tests prove, on a tiny
// synthetic model (no checkpoint/torch), that routing a position through the
// embedding-injection path is bit-identical to the token-id path. That is the
// "provably inert when no image is present" gate: the image-embedding seam can't
// perturb text-only behavior.

// tinyLlamaModel builds a tiny but real llama (generic forward path — the path
// Gemma 3, the first multimodal target, also uses) with deterministic weights.
func tinyLlamaModel(t *testing.T) (*Model, int) {
	t.Helper()
	const hidden, heads, headDim, inter, vocab, layers = 8, 2, 4, 16, 16, 2
	qDim := heads * headDim // 8
	base := t.TempDir()
	cfg := `{"model_type":"llama","vocab_size":16,"hidden_size":8,"num_hidden_layers":2,
		"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"intermediate_size":16,
		"max_position_embeddings":128,"rms_norm_eps":1e-6,"rope_theta":10000}`
	if err := os.WriteFile(filepath.Join(base, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	fill := func(n int) []float32 { // deterministic, varied, small magnitude
		d := make([]float32, n)
		for i := range d {
			d[i] = float32(i%7)*0.1 - 0.3
		}
		return d
	}
	ts := map[string]stTensor{
		"model.embed_tokens.weight": {[]int{vocab, hidden}, fill(vocab * hidden)},
		"model.norm.weight":         {[]int{hidden}, fill(hidden)},
		"lm_head.weight":            {[]int{vocab, hidden}, fill(vocab * hidden)},
	}
	for l := range layers {
		p := fmt.Sprintf("model.layers.%d.", l)
		ts[p+"self_attn.q_proj.weight"] = stTensor{[]int{qDim, hidden}, fill(qDim * hidden)}
		ts[p+"self_attn.k_proj.weight"] = stTensor{[]int{qDim, hidden}, fill(qDim * hidden)}
		ts[p+"self_attn.v_proj.weight"] = stTensor{[]int{qDim, hidden}, fill(qDim * hidden)}
		ts[p+"self_attn.o_proj.weight"] = stTensor{[]int{hidden, qDim}, fill(hidden * qDim)}
		ts[p+"mlp.gate_proj.weight"] = stTensor{[]int{inter, hidden}, fill(inter * hidden)}
		ts[p+"mlp.up_proj.weight"] = stTensor{[]int{inter, hidden}, fill(inter * hidden)}
		ts[p+"mlp.down_proj.weight"] = stTensor{[]int{hidden, inter}, fill(hidden * inter)}
		ts[p+"input_layernorm.weight"] = stTensor{[]int{hidden}, fill(hidden)}
		ts[p+"post_attention_layernorm.weight"] = stTensor{[]int{hidden}, fill(hidden)}
	}
	writeSafetensors(t, filepath.Join(base, "model.safetensors"), ts)

	w, err := loadWeights(base, quantNone, false, nil)
	if err != nil {
		t.Fatalf("loadWeights: %v", err)
	}
	m, err := NewModel(w, "cpu")
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	return m, hidden
}

// TestMultimodal_embedInjectionMatchesIDPath: forwardFromEmbed(embedToken(id))
// must equal forward(id) bit-for-bit — the embed-by-vector seam reproduces the
// token-id path exactly, so injecting image embeddings can't perturb text.
func TestMultimodal_embedInjectionMatchesIDPath(t *testing.T) {
	m, hidden := tinyLlamaModel(t)
	for _, id := range []int{0, 1, 5, 9, 15} {
		idCache := m.NewCache(1)
		want, err := m.forward(id, idCache)
		if err != nil {
			t.Fatalf("forward(%d): %v", id, err)
		}
		want = append([]float32(nil), want...) // forward returns cache scratch; copy before reuse

		embCache := m.NewCache(1)
		e := make([]float32, hidden)
		m.embedToken(id, e)
		got, err := m.forwardFromEmbed(e, embCache)
		if err != nil {
			t.Fatalf("forwardFromEmbed(id=%d): %v", id, err)
		}
		if len(got) != len(want) {
			t.Fatalf("id=%d: logit len %d != %d", id, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("id=%d: logits differ at %d: embed-path %v vs id-path %v (must be bit-identical)", id, i, got[i], want[i])
			}
		}
	}
}

// TestMultimodal_imageBlockMask: the bidirectional image-block mask seam.
// Comparing the batched prefill with a block set vs no block (same code path, the
// only difference being attendHi): a position OUTSIDE the block is bit-identical
// (the seam is inert for text), and the first IN-block position changes (it now
// attends to the block's future tokens — the mask is actually wired).
func TestMultimodal_imageBlockMask(t *testing.T) {
	m, _ := tinyLlamaModel(t)
	seq := []int{3, 7, 1, 12, 5} // K=5 > 1 → the batched attendBatchedHeads path

	baseCache := m.NewCache(len(seq))
	baseline, err := m.forwardN(seq, baseCache) // no image blocks → fully causal
	if err != nil {
		t.Fatalf("forwardN baseline: %v", err)
	}

	maskCache := m.NewCache(len(seq))
	maskCache.SetImageBlocks([][2]int{{1, 4}}) // positions 1,2,3 attend bidirectionally
	masked, err := m.forwardN(seq, maskCache)
	if err != nil {
		t.Fatalf("forwardN masked: %v", err)
	}

	// Position 0 is before the block and causal — its attention and residual never
	// depend on positions 1..4, so it must be bit-identical (seam inert off-block).
	for i := range baseline[0] {
		if masked[0][i] != baseline[0][i] {
			t.Fatalf("pos 0 (outside block) changed at logit %d: %v vs %v — mask not inert for text", i, masked[0][i], baseline[0][i])
		}
	}
	// Position 1 is the first in-block position: bidirectional attention now sees
	// the block's future tokens (2,3), so its logits must differ from causal.
	same := true
	for i := range baseline[1] {
		if masked[1][i] != baseline[1][i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("pos 1 (in bidirectional block) identical to causal — the mask had no effect")
	}
}

// TestMultimodal_embedInjectionSequence: a multi-position decode driven through
// the embed-injection path (every token fed as its own embedding) reproduces the
// id-path token-for-token — the seam composes across positions / a growing KV
// cache, not just at position 0.
func TestMultimodal_embedInjectionSequence(t *testing.T) {
	m, hidden := tinyLlamaModel(t)
	seq := []int{3, 7, 1, 12, 0}

	idCache := m.NewCache(len(seq))
	var idLogits [][]float32
	for _, id := range seq {
		l, err := m.forward(id, idCache)
		if err != nil {
			t.Fatalf("forward: %v", err)
		}
		idLogits = append(idLogits, append([]float32(nil), l...))
	}

	embCache := m.NewCache(len(seq))
	e := make([]float32, hidden)
	for s, id := range seq {
		m.embedToken(id, e)
		l, err := m.forwardFromEmbed(e, embCache)
		if err != nil {
			t.Fatalf("forwardFromEmbed: %v", err)
		}
		for i := range l {
			if l[i] != idLogits[s][i] {
				t.Fatalf("step %d (id=%d): embed-path diverges from id-path at logit %d", s, id, i)
			}
		}
	}
}

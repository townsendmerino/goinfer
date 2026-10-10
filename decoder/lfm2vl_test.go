package decoder

import (
	"context"
	"errors"
	"io/fs"
	"math/rand"
	"os"
	"slices"
	"testing"
)

// S10, LFM2.5-VL (docs/tasks/task-multimodal-support-2026-10.md, "S10, LFM2.5-VL"): the causal image entry on the tracked
// lfm2-tiny decoder. lfm2 has no batched pass, so prefillHiddenLFM2VL runs the spliced rows one at a time through
// runLayersLFM2FromEmbed. Three properties:
//   - with the text embeddings themselves spliced in as the "image" rows, it equals the plain per-token forward over the
//     ids bit for bit (the embedding entry is the id entry);
//   - with real features, it equals a hand splice of the <image> positions, and a planted splice that also covers the
//     tile marker between the runs differs;
//   - GenerateVLCausalSpans' first token is the argmax of that prefill.
// The full model against HF is G-S10l-c, on nobara.

func loadLfm2Tiny(t *testing.T) *Model {
	t.Helper()
	if _, err := os.Stat("../testdata/lfm2-tiny/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no lfm2-tiny checkpoint")
	}
	m, err := Load("../testdata/lfm2-tiny", Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	if m.w.arch.lfm2 == nil {
		t.Fatalf("lfm2-tiny resolved to %q, not lfm2", m.w.arch.Name)
	}
	return m
}

func TestLfm2VLCausalSpans_tiny(t *testing.T) {
	m := loadLfm2Tiny(t)
	ctx := context.Background()
	hidden := m.w.arch.HiddenDim
	// start, a marker, a 3-token tile, a marker, a 2-token thumbnail, end, text.
	const img, start, mark, thumb, end = 40, 41, 42, 43, 44
	ids := []int{1, 5, start, mark, img, img, img, thumb, img, img, end, 20, 21, 22}
	spans := []ImageSpan{{Pos: 4, Len: 3, Hash: 9}, {Pos: 8, Len: 2, Hash: 9}}

	// 1. The text embeddings as the image rows: the plain token forward, bit for bit.
	emb := m.embedN(ids)
	var self []float32
	for _, s := range spans {
		self = append(self, emb[s.Pos*hidden:(s.Pos+s.Len)*hidden]...)
	}
	got, err := m.prefillHiddenVLSpans(ctx, ids, spans, self, m.NewCache(len(ids)), true)
	if err != nil {
		t.Fatalf("prefillHiddenVLSpans: %v", err)
	}
	cache := m.NewCache(len(ids))
	var plain []float32
	for _, id := range ids {
		h, err := m.runLayersLFM2(id, cache)
		if err != nil {
			t.Fatal(err)
		}
		normalize(m.w.arch, h, m.w.FinalNorm, m.w.FinalNormBias, hidden)
		plain = append(plain, h...)
	}
	if !slices.Equal(got, plain) {
		t.Errorf("the embedding entry with the text's own rows differs from the token forward")
	}

	// 2. Real features against a hand splice; a planted splice over the marker differs.
	r := rand.New(rand.NewSource(13))
	feats := make([]float32, 5*hidden)
	for i := range feats {
		feats[i] = float32(r.NormFloat64()) * 0.5
	}
	got, err = m.prefillHiddenVLSpans(ctx, ids, spans, feats, m.NewCache(len(ids)), true)
	if err != nil {
		t.Fatal(err)
	}
	hand := func(over []int) []float32 {
		h := m.embedN(ids)
		for k, p := range over {
			copy(h[p*hidden:(p+1)*hidden], feats[min(k, 4)*hidden:(min(k, 4)+1)*hidden])
		}
		c := m.NewCache(len(ids))
		for i := range ids {
			out, err := m.runLayersLFM2FromEmbed(h[i*hidden:(i+1)*hidden], c)
			if err != nil {
				t.Fatal(err)
			}
			normalize(m.w.arch, out, m.w.FinalNorm, m.w.FinalNormBias, hidden)
		}
		return h
	}
	if want := hand([]int{4, 5, 6, 8, 9}); !slices.Equal(got, want) {
		t.Errorf("the LFM2-VL prefill differs from a hand splice of the <image> positions")
	}
	if planted := hand([]int{4, 5, 6, 7, 8, 9}); slices.Equal(got, planted) {
		t.Errorf("planted (features over the thumbnail marker too) equals the production path: the check sees nothing")
	}

	// 3. The generate entry decodes from the same prefill.
	logits := m.lmHeadN(got[(len(ids)-1)*hidden:], 1)
	stream, g := m.GenerateVLCausalSpans(ctx, ids, spans, func() ([]float32, error) { return feats, nil }, 2, SamplingParams{})
	var toks []int
	for tok := range stream {
		toks = append(toks, tok)
	}
	if err := g.Err(); err != nil {
		t.Fatalf("GenerateVLCausalSpans: %v", err)
	}
	if want := argmaxF32(logits); len(toks) == 0 || toks[0] != want {
		t.Errorf("GenerateVLCausalSpans produced %v, want first %d (the prefill's argmax)", toks, want)
	}
}

// TestLfm2FFNDim pins HF's Lfm2MLP width rule (S10: LFM2-VL-1.6B's text_config has block_auto_adjust_ff_dim, 12288 ->
// 8192, and a load at the stated width was refused by the checkpoint's own shapes).
func TestLfm2FFNDim(t *testing.T) {
	one := 1.0
	for _, c := range []struct {
		inter, multipleOf int
		auto              bool
		mult              *float64
		want              int
	}{
		{10752, 256, false, &one, 10752}, // LFM2.5-2.6B: stated
		{12288, 256, true, &one, 8192},   // LFM2-1.2B (LFM2.5-VL-1.6B's text_config)
		{1000, 256, true, &one, 768},     // rounded up to the multiple
		{1000, 256, true, nil, 666},      // no multiplier: HF does not round
	} {
		cfg := &Config{IntermediateDim: c.inter, BlockAutoAdjustFFDim: c.auto, BlockFFNDimMultiplier: c.mult, BlockMultipleOf: c.multipleOf}
		if got := lfm2FFNDim(cfg); got != c.want {
			t.Errorf("intermediate %d auto %v mult %v multiple_of %d: %d, want %d", c.inter, c.auto, c.mult, c.multipleOf, got, c.want)
		}
	}
}

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

// S10, Ministral 3 (docs/tasks/task-multimodal-support-2026-10.md, "S10, Ministral 3 (Pixtral)"): the causal span entry
// on the tiny ministral3 decoder. Pixtral's features replace the [IMG] positions only, one span per merged row, with
// [IMG_BREAK]/[IMG_END] left as text, and the image tokens attend causally (no image block). Three properties, each with
// a planted defect that must break it:
//   - the production path equals a hand splice of the rows into the text embeddings, run causally (planted: the
//     features also written over the break between the rows);
//   - changing an image row's features leaves every earlier position's hidden state bit-identical (planted: the
//     bidirectional image block, which the non-causal entry sets);
//   - GenerateVLCausalSpans' first token is the argmax of those prefill logits.
// The full model against HF is G-S10m-c, on nobara.

func loadMinistral3Tiny(t *testing.T) *Model {
	t.Helper()
	if _, err := os.Stat(ministral3ModelDir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no Ministral3 checkpoint at %s", ministral3ModelDir)
	}
	m, err := Load(ministral3ModelDir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// pixtralCausalCase is a prompt with two merged rows of three [IMG] positions, a break between them and an end after.
func pixtralCausalCase(m *Model) (ids []int, spans []ImageSpan, feats []float32) {
	const img, brk, end = 10, 12, 13
	ids = []int{1, 3, 5, 7, img, img, img, brk, img, img, img, end, 20, 21, 22}
	spans = []ImageSpan{{Pos: 4, Len: 3, Hash: 9}, {Pos: 8, Len: 3, Hash: 9}}
	hidden := m.w.arch.HiddenDim
	r := rand.New(rand.NewSource(11))
	feats = make([]float32, 6*hidden)
	for i := range feats {
		feats[i] = float32(r.NormFloat64()) * 0.5
	}
	return ids, spans, feats
}

func TestPixtralCausalSpans_tiny(t *testing.T) {
	m := loadMinistral3Tiny(t)
	ctx := context.Background()
	hidden := m.w.arch.HiddenDim
	ids, spans, feats := pixtralCausalCase(m)

	got, err := m.prefillHiddenVLSpans(ctx, ids, spans, feats, m.NewCache(len(ids)), true)
	if err != nil {
		t.Fatalf("prefillHiddenVLSpans: %v", err)
	}
	// A hand splice: positions 4-6 and 8-10 take the six rows in order; 7 (the break) and 11 (the end) stay text.
	handSplice := func(over []int) []float32 {
		h := m.embedN(ids)
		row := 0
		for _, p := range over {
			copy(h[p*hidden:(p+1)*hidden], feats[row*hidden:(row+1)*hidden])
			row = min(row+1, 5)
		}
		hN, err := m.runLayersFromEmbedN(ctx, h, m.NewCache(len(ids)), m.cpuFastAttention())
		if err != nil {
			t.Fatalf("runLayersFromEmbedN: %v", err)
		}
		return hN
	}
	if want := handSplice([]int{4, 5, 6, 8, 9, 10}); !slices.Equal(got, want) {
		t.Errorf("the causal span prefill differs from a hand splice of the [IMG] positions")
	}
	if planted := handSplice([]int{4, 5, 6, 7, 8, 9, 10}); slices.Equal(got, planted) {
		t.Errorf("planted (features over the break too) equals the production path: the check sees nothing")
	}

	// Causality: the last image row changed, every earlier position unchanged.
	feats2 := slices.Clone(feats)
	for i := 5 * hidden; i < 6*hidden; i++ {
		feats2[i] += 1
	}
	at := (spans[1].Pos + spans[1].Len - 1) * hidden // position 10, the last image row
	caus2, err := m.prefillHiddenVLSpans(ctx, ids, spans, feats2, m.NewCache(len(ids)), true)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got[:at], caus2[:at]) {
		t.Errorf("changing the last image row changed an earlier position: the image tokens are not causal")
	}
	if slices.Equal(got[at:], caus2[at:]) {
		t.Errorf("changing the last image row changed nothing at or after it: the features are not reaching the model")
	}
	bidi1, err := m.prefillHiddenVLSpans(ctx, ids, spans, feats, m.NewCache(len(ids)), false)
	if err != nil {
		t.Fatal(err)
	}
	bidi2, err := m.prefillHiddenVLSpans(ctx, ids, spans, feats2, m.NewCache(len(ids)), false)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(bidi1[:at], bidi2[:at]) {
		t.Errorf("planted (the bidirectional image block) left the earlier positions unchanged: the causality check sees nothing")
	}

	// The generate entry decodes from the same prefill.
	logits := m.lmHeadN(got[(len(ids)-1)*hidden:], 1)
	stream, g := m.GenerateVLCausalSpans(ctx, ids, spans, func() ([]float32, error) { return feats, nil }, 1, SamplingParams{})
	var toks []int
	for tok := range stream {
		toks = append(toks, tok)
	}
	if err := g.Err(); err != nil {
		t.Fatalf("GenerateVLCausalSpans: %v", err)
	}
	if want := argmaxF32(logits); len(toks) != 1 || toks[0] != want {
		t.Errorf("GenerateVLCausalSpans produced %v, want [%d] (the prefill's argmax)", toks, want)
	}
}

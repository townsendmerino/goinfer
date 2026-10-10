package decoder

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"math"
	"os"
	"slices"
	"strings"
	"testing"
)

// S11 (docs/tasks/task-multimodal-support-2026-10.md, "S11, plan and gates"): two images in one prompt, against the
// two-image goldens scripts/pin_two_images_tiny.py pinned from HF (transformers 5.15.0). Each layout is graded at the
// family's existing single-image bar, at the last position and at every position; each planted defect must miss it.

// twoImageLayout is one layout of a two-image golden.
type twoImageLayout struct {
	InputIDs           []int       `json:"input_ids"`
	Spans              [][2]int    `json:"spans"` // [start, n]
	Logits             [][]float32 `json:"logits"`
	SecondCausalLogits [][]float32 `json:"second_causal_logits"`
	SecondCausalDiff   float64     `json:"second_causal_max_abs_diff"`
	Continuation       []int       `json:"continuation_ids"`
	// Qwen only
	GridTHW       [][3]int    `json:"grid_thw"`
	ImageFeatures [][]float32 `json:"image_features"`
	PositionIDs   [][]int     `json:"position_ids"`
	RopeDelta     int         `json:"rope_delta"`
}

// twoImageGolden is the file: per-image features (Gemma: top level; Qwen: per layout) and the two layouts.
type twoImageGolden struct {
	ImageFeatures [][]float32               `json:"image_features"`
	Layouts       map[string]twoImageLayout `json:"layouts"`
}

func loadTwoImageGolden(t *testing.T, golden, ckpt string) twoImageGolden {
	t.Helper()
	if _, err := os.Stat(ckpt); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no checkpoint %s", ckpt)
	}
	var g twoImageGolden
	raw, err := readTwoImageGolden(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden %s: run scripts/pin_two_images_tiny.py", golden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(g.Layouts) != 2 {
		t.Fatalf("golden has %d layouts, want interleaved and adjacent", len(g.Layouts))
	}
	return g
}

// readTwoImageGolden reads a golden, gunzipping a .gz one (the untagged twin of the realckpt-tagged readGolden).
func readTwoImageGolden(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if !strings.HasSuffix(path, ".gz") {
		return io.ReadAll(f)
	}
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()
	return io.ReadAll(gz)
}

// spansOf is a layout's spans (no reuse claim).
func (l twoImageLayout) spansOf() []ImageSpan {
	s := make([]ImageSpan, len(l.Spans))
	for i, sp := range l.Spans {
		s[i] = ImageSpan{Pos: sp[0], Len: sp[1]}
	}
	return s
}

// gradeAllPositions compares every position's logits with the golden's: the worst cosine, and whether the last
// position's argmax matches. pass is the family's single-image bar (cosine >= bar and the last argmax exact) held at
// the last position and at every position.
func gradeAllPositions(got, want [][]float32, bar float64) (worst float64, lastArgOK, pass bool) {
	worst = 1
	for i := range want {
		worst = math.Min(worst, logitCosine(got[i], want[i]))
	}
	n := len(want) - 1
	lastArgOK = argmax(got[n]) == argmax(want[n])
	return worst, lastArgOK, worst >= bar && lastArgOK
}

// rowsOf splits a flat [n*V] logits slab into rows.
func rowsOf(flat []float32, n int) [][]float32 {
	v := len(flat) / n
	r := make([][]float32, n)
	for i := range r {
		r[i] = flat[i*v : (i+1)*v]
	}
	return r
}

func concatFeats(f ...[]float32) []float32 {
	var out []float32
	for _, x := range f {
		out = append(out, x...)
	}
	return out
}

// TestTwoImages_gemma3 (G-S11a, G-S11b): Gemma 3's two-image prefill, each image its own bidirectional block.
func TestTwoImages_gemma3(t *testing.T) {
	const ckpt = "../testdata/gemma3-vl-tiny"
	g := loadTwoImageGolden(t, "../testdata/gemma3_vl_tiny_two_images_golden.json", ckpt)
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	hidden := m.w.arch.HiddenDim
	const bar = 0.99 // TestGemma3VL_imageParity's (the batched-attention floor), with its exact last argmax
	ctx := context.Background()
	all := func(ids []int, spans []ImageSpan, feats []float32) [][]float32 {
		hN, err := m.prefillHiddenVLSpans(ctx, ids, spans, feats, m.NewCache(len(ids)), false)
		if err != nil {
			t.Fatalf("prefillHiddenVLSpans: %v", err)
		}
		return rowsOf(m.lmHeadN(hN, len(ids)), len(ids))
	}
	for _, name := range []string{"interleaved", "adjacent"} {
		l := g.Layouts[name]
		spans := l.spansOf()
		feats := concatFeats(g.ImageFeatures[0], g.ImageFeatures[1])
		worst, argOK, pass := gradeAllPositions(all(l.InputIDs, spans, feats), l.Logits, bar)
		t.Logf("%s: two images, worst cosine over %d positions %.6f, last argmax ok %v", name, len(l.InputIDs), worst, argOK)
		if !pass {
			t.Errorf("%s: two-image prefill misses the bar: worst cosine %.6f (>= %.2f), last argmax ok %v", name, worst, bar, argOK)
		}
		// G-S11b (2): the two images' features swapped.
		sw, swArg, swPass := gradeAllPositions(all(l.InputIDs, spans, concatFeats(g.ImageFeatures[1], g.ImageFeatures[0])), l.Logits, bar)
		t.Logf("%s: planted defect, features swapped: worst %.6f, last argmax ok %v", name, sw, swArg)
		if swPass {
			t.Errorf("%s: BLIND: the swapped images still clear the bar (worst %.6f)", name, sw)
		}
		// G-S11b (3): the second block left causal (both images spliced, only the first registered as a block). The
		// mutant is built here on purpose; it must also reproduce HF's own second-block-causal forward.
		h := m.embedN(l.InputIDs)
		spliceImageSpans(h, spans, feats, hidden)
		c := m.NewCache(len(l.InputIDs))
		c.SetImageBlocks(imageSpanBlocks(spans[:1]))
		hN, err := m.runLayersFromEmbedN(ctx, h, c, m.cpuFastAttention())
		if err != nil {
			t.Fatal(err)
		}
		bad := rowsOf(m.lmHeadN(hN, len(l.InputIDs)), len(l.InputIDs))
		bw, bArg, bPass := gradeAllPositions(bad, l.Logits, bar)
		hw, _, hPass := gradeAllPositions(bad, l.SecondCausalLogits, bar)
		t.Logf("%s: planted defect, second block causal: worst %.6f against the golden (HF's own defect max|diff| %.4f), last argmax ok %v; against HF's defect forward worst %.6f", name, bw, l.SecondCausalDiff, bArg, hw)
		if bPass {
			t.Errorf("%s: BLIND: the second block left causal still clears the bar (worst %.6f)", name, bw)
		}
		if !hPass {
			t.Errorf("%s: the second-block-causal mutant does not reproduce HF's own defect forward (worst %.6f)", name, hw)
		}
	}
}

// qwenTwoImages runs G-S11a/G-S11b for a Qwen-family tiny fixture: the m-RoPE positions exact, the last position at the
// family's bar with its argmax exact, and the greedy continuation exact (decode past two images resumes at seqPos+delta).
// Planted defects: (1) the adjacent same-size pair positioned as one image's two temporal halves (image B's positions
// are A's at t+1 instead of a fresh base after A: the pairing goinfer must never do); (2) the two images' features
// swapped. Each must miss the bar.
// blindOK names the "<layout>: <defect>" pairs this fixture is known to be blind to; those log BLIND instead of failing
// (the registration: a blind fixture says so, and the other fixtures and G-S11e carry the defect).
func qwenTwoImages(t *testing.T, ckpt, golden string, bar float64, blindOK map[string]bool) {
	g := loadTwoImageGolden(t, golden, ckpt)
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	const merge = 2 // both tiny fixtures' spatial_merge_size
	var raw struct {
		ImageToken int `json:"image_token_id"`
	}
	b, _ := readTwoImageGolden(golden)
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	run := func(l twoImageLayout, feats []float32, pos [][3]int) (last []float32, cont []int) {
		c := m.NewCache(len(l.InputIDs) + len(l.Continuation))
		lg, err := m.prefillLogitsQwenVLSpans(ctx, l.InputIDs, l.spansOf(), feats, pos, c)
		if err != nil {
			t.Fatalf("prefillLogitsQwenVLSpans: %v", err)
		}
		last = append([]float32(nil), lg...)
		for range l.Continuation {
			next := argmax(lg)
			cont = append(cont, next)
			if lg, err = m.forward(next, c); err != nil {
				t.Fatal(err)
			}
		}
		return last, cont
	}
	for _, name := range []string{"interleaved", "adjacent"} {
		l := g.Layouts[name]
		pos, err := mropePositions(l.InputIDs, raw.ImageToken, l.GridTHW, merge)
		if err != nil {
			t.Fatalf("%s: mropePositions: %v", name, err)
		}
		samePos := func(p [][3]int) bool {
			ok := len(l.PositionIDs) == 3 && len(p) == len(l.PositionIDs[0])
			for i := range p {
				for a := 0; a < 3 && ok; a++ {
					ok = p[i][a] == l.PositionIDs[a][i]
				}
			}
			return ok
		}
		if !samePos(pos) {
			t.Errorf("%s: m-RoPE positions differ from HF's", name)
		}
		if d := mropeDelta(pos, len(l.InputIDs)); d != l.RopeDelta {
			t.Errorf("%s: rope delta %d, HF %d", name, d, l.RopeDelta)
		}
		want := l.Logits[len(l.Logits)-1]
		feats := concatFeats(l.ImageFeatures[0], l.ImageFeatures[1])
		last, cont := run(l, feats, pos)
		cos := logitCosine(last, want)
		t.Logf("%s: two images (grids %v): last cosine %.6f, argmax %d (HF %d), continuation %v (HF %v)", name, l.GridTHW, cos, argmax(last), argmax(want), cont, l.Continuation)
		if cos < bar || argmax(last) != argmax(want) || !slices.Equal(cont, l.Continuation) {
			t.Errorf("%s: two-image prefill misses the bar (cosine %.6f >= %.4f, argmax and continuation exact)", name, cos, bar)
		}
		// The whole bar, positions included: a defect clears it only if its positions are HF's too.
		red := func(what string, p [][3]int, last []float32, cont []int) {
			c := logitCosine(last, want)
			t.Logf("%s: planted defect, %s: positions HF's %v, last cosine %.6f, argmax %d (HF %d), continuation %v", name, what, samePos(p), c, argmax(last), argmax(want), cont)
			if samePos(p) && c >= bar && argmax(last) == argmax(want) && slices.Equal(cont, l.Continuation) {
				if blindOK[name+": "+what] {
					t.Logf("%s: BLIND, known: this fixture cannot see %q at its bar; the other fixtures and G-S11e carry it", name, what)
				} else {
					t.Errorf("%s: BLIND: %s still clears the bar", name, what)
				}
			}
		}
		lsw, csw := run(l, concatFeats(l.ImageFeatures[1], l.ImageFeatures[0]), pos)
		red("features swapped", pos, lsw, csw)
		if name == "adjacent" {
			// (1) pairing: image B positioned as A's second temporal half.
			paired := append([][3]int(nil), pos...)
			sp := l.spansOf()
			for k := 0; k < sp[1].Len; k++ {
				a := pos[sp[0].Pos+k]
				paired[sp[1].Pos+k] = [3]int{a[0] + 1, a[1], a[2]}
			}
			lp, cp := run(l, feats, paired)
			red("the pair positioned as one image's temporal halves", paired, lp, cp)
		}
	}
}

// TestTwoImages_qwen25vl (G-S11a, G-S11b): Qwen2.5-VL, two images, at its image golden's bar (cosine >= 0.99).
func TestTwoImages_qwen25vl(t *testing.T) {
	// Its rope theta is the released one, so a last-position bar of 0.99 barely sees what the image rows carry (the
	// interleaved swap reads 0.9933 there). The Qwen3.5 fixture (theta 10) and Gemma 3 catch it.
	qwenTwoImages(t, "../testdata/qwen25vl-tiny", "../testdata/qwen25vl_tiny_two_images_golden.json.gz", 0.99,
		map[string]bool{"interleaved: features swapped": true})
}

// TestTwoImages_qwen35vl (G-S11a, G-S11b): the Qwen3.5 hybrid, two images, at its image golden's bar (cosine >= 0.9999).
func TestTwoImages_qwen35vl(t *testing.T) {
	qwenTwoImages(t, "../testdata/qwen35vl-tiny", "../testdata/qwen35vl_tiny_two_images_golden.json", 0.9999, nil)
}

// TestTwoImages_gemma4Bidir (G-S11a, G-S11b): a bidirectional ("vision") Gemma 4, two images, each its own block, at the
// single-image golden's bar (last-logit cosine >= 0.999, argmax exact), plus the greedy continuation exact. Planted:
// the images' features swapped; the second block left causal, which must also reproduce HF's own forward for it.
func TestTwoImages_gemma4Bidir(t *testing.T) {
	const ckpt = "../testdata/gemma4-vl-bidir-tiny"
	g := loadTwoImageGolden(t, "../testdata/gemma4_vl_bidir_tiny_two_images_golden.json", ckpt)
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	const bar = 0.999 // TestGemma4VLBidir_imageParity's
	hidden := m.w.arch.HiddenDim
	ctx := context.Background()
	for _, name := range []string{"interleaved", "adjacent"} {
		l := g.Layouts[name]
		spans := l.spansOf()
		feats := concatFeats(g.ImageFeatures[0], g.ImageFeatures[1])
		want := l.Logits[len(l.Logits)-1]
		c := m.NewCache(len(l.InputIDs) + len(l.Continuation))
		lg, err := m.prefillLogitsGemma4VLBidirectionalSpans(ctx, l.InputIDs, feats, spans, c)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		last := append([]float32(nil), lg...)
		var cont []int
		for range l.Continuation {
			next := argmax(lg)
			cont = append(cont, next)
			if lg, err = m.forward(next, c); err != nil {
				t.Fatal(err)
			}
		}
		cos := logitCosine(last, want)
		t.Logf("%s: two images: last cosine %.6f, argmax %d (HF %d), continuation %v (HF %v)", name, cos, argmax(last), argmax(want), cont, l.Continuation)
		if cos < bar || argmax(last) != argmax(want) || !slices.Equal(cont, l.Continuation) {
			t.Errorf("%s: two-image prefill misses the bar (cosine %.6f >= %.3f, argmax and continuation exact)", name, cos, bar)
		}
		sw, err := m.prefillLogitsGemma4VLBidirectionalSpans(ctx, l.InputIDs, concatFeats(g.ImageFeatures[1], g.ImageFeatures[0]), spans, m.NewCache(len(l.InputIDs)))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: planted defect, features swapped: last cosine %.6f, argmax %d", name, logitCosine(sw, want), argmax(sw))
		if logitCosine(sw, want) >= bar && argmax(sw) == argmax(want) {
			t.Errorf("%s: BLIND: the swapped images still clear the bar", name)
		}
		// The second block left causal: both images spliced, only the first a block.
		h := m.embedN(l.InputIDs)
		spliceImageSpans(h, spans, feats, hidden)
		bc := m.NewCache(len(l.InputIDs))
		hLast, err := m.runLayersGemma4FromEmbedN(ctx, h, l.InputIDs, spans[:1], bc)
		if err != nil {
			t.Fatal(err)
		}
		bad := m.logitsFromHidden(hLast, bc)
		hfBad := l.SecondCausalLogits[len(l.SecondCausalLogits)-1]
		t.Logf("%s: planted defect, second block causal: last cosine %.6f against the golden, argmax %d; against HF's own defect forward %.6f (HF max|diff| %.4f)", name, logitCosine(bad, want), argmax(bad), logitCosine(bad, hfBad), l.SecondCausalDiff)
		if logitCosine(bad, want) >= bar && argmax(bad) == argmax(want) {
			t.Errorf("%s: BLIND: the second block left causal still clears the bar", name)
		}
		if logitCosine(bad, hfBad) < bar {
			t.Errorf("%s: the second-block-causal mutant does not reproduce HF's own defect forward (%.6f)", name, logitCosine(bad, hfBad))
		}
	}
}

// TestTwoImages_reuse (G-S11d): resident prefix reuse with two images. After a committed two-image turn, the same two
// images reuse the whole prompt (the tower is not run again); a changed second image reuses no further than the second
// block's start, a changed first image no further than the first block's start, and either change runs the tower.
func TestTwoImages_reuse(t *testing.T) {
	const ckpt = "../testdata/gemma3-vl-tiny"
	g := loadTwoImageGolden(t, "../testdata/gemma3_vl_tiny_two_images_golden.json", ckpt)
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	rf := &fakeResident{vocab: m.w.arch.VocabSize}
	m.resident = rf
	l := g.Layouts["interleaved"]
	towerCalls := 0
	features := func() ([]float32, error) {
		towerCalls++
		return concatFeats(g.ImageFeatures[0], g.ImageFeatures[1]), nil
	}
	spansWith := func(hashA, hashB uint64) []ImageSpan {
		s := l.spansOf()
		s[0].Hash, s[1].Hash = hashA, hashB
		return s
	}
	run := func(prompt []int, spans []ImageSpan) ([]int, *Generation) {
		stream, gen := m.GenerateVLSpans(context.Background(), prompt, spans, features, 3, SamplingParams{Temperature: 0})
		var out []int
		for id := range stream {
			out = append(out, id)
		}
		if err := gen.Err(); err != nil {
			t.Fatalf("GenerateVLSpans: %v", err)
		}
		return out, gen
	}
	spans := spansWith(11, 22)
	lastEnd := spans[1].Pos + spans[1].Len
	gen1, g1 := run(l.InputIDs, spans)
	if towerCalls != 1 || g1.PrefillReused != 0 {
		t.Fatalf("turn 1: tower calls %d (want 1), PrefillReused %d (want 0)", towerCalls, g1.PrefillReused)
	}
	if len(m.resImgBlocks) != 2 {
		t.Fatalf("turn 1 committed %d image blocks, want 2", len(m.resImgBlocks))
	}
	prompt2 := append(append(append([]int{}, l.InputIDs...), gen1...), l.InputIDs[len(l.InputIDs)-1])
	// The bound each claim set reaches, read before any turn changes the record.
	for _, c := range []struct {
		name    string
		spans   []ImageSpan
		within  int // reuse must not reach past this
		atLeast int
	}{
		{"same two images", spansWith(11, 22), len(prompt2), lastEnd},
		{"second image changed", spansWith(11, 33), spans[1].Pos, 0},
		{"first image changed", spansWith(44, 22), spans[0].Pos, 0},
	} {
		got := m.residentReuseLen(prompt2, residentImageClaimsOf(c.spans), nil)
		t.Logf("%s: reuse reaches %d (bound %d, at least %d)", c.name, got, c.within, c.atLeast)
		if got > c.within || got < c.atLeast {
			t.Errorf("%s: reuse reaches %d, want within [%d, %d]", c.name, got, c.atLeast, c.within)
		}
	}
	// End to end: the same images reuse in full without the tower; a changed second image runs it.
	_, g2 := run(prompt2, spansWith(11, 22))
	if towerCalls != 1 || g2.PrefillReused < lastEnd {
		t.Errorf("same two images: tower calls %d (want still 1), PrefillReused %d (want >= %d)", towerCalls, g2.PrefillReused, lastEnd)
	}
	gen3, _ := run(l.InputIDs, spansWith(11, 22)) // a fresh committed record of the original prompt
	_ = gen3
	before := towerCalls
	prompt3 := append(append(append([]int{}, l.InputIDs...), gen3...), l.InputIDs[len(l.InputIDs)-1])
	_, g4 := run(prompt3, spansWith(11, 33))
	if towerCalls != before+1 || g4.PrefillReused != 0 {
		t.Errorf("second image changed: tower calls %d (want %d), PrefillReused %d (want 0: no full reuse)", towerCalls, before+1, g4.PrefillReused)
	}
}

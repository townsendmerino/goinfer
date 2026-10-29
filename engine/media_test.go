package engine

import (
	"context"
	"reflect"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// mediaCall is one generator call a fakeMedia saw.
type mediaCall struct {
	name       string
	ids        []int
	spans      []decoder.ImageSpan
	pos, n     int
	grids      [][3]int
	merge, tok int
	max        int
	temp       float32
	rows       []float32
	deep       [][]float32
}

type fakeMedia struct {
	hidden int
	calls  []mediaCall
}

func (f *fakeMedia) Config() *decoder.Config { return &decoder.Config{HiddenDim: f.hidden} }

func (f *fakeMedia) record(c mediaCall, features func() ([]float32, error)) (<-chan int, *decoder.Generation) {
	if features != nil {
		c.rows, _ = features()
	}
	f.calls = append(f.calls, c)
	ch := make(chan int)
	close(ch)
	return ch, &decoder.Generation{}
}

func (f *fakeMedia) GenerateAudio(_ context.Context, ids []int, pos, n int, feat func() ([]float32, error), max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation) {
	return f.record(mediaCall{name: "audio", ids: ids, pos: pos, n: n, max: max, temp: sp.Temperature}, feat)
}

func (f *fakeMedia) GenerateGemma4VLSpans(_ context.Context, ids []int, spans []decoder.ImageSpan, feat func() ([]float32, error), max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation) {
	return f.record(mediaCall{name: "gemma4", ids: ids, spans: spans, max: max, temp: sp.Temperature}, feat)
}

func (f *fakeMedia) GenerateQwenVLDeepstackSpans(_ context.Context, ids []int, spans []decoder.ImageSpan, feat func() ([]float32, [][]float32, error), grids [][3]int, merge, tok, max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation) {
	rows, deep, _ := feat()
	c := mediaCall{name: "qwen", ids: ids, spans: spans, grids: grids, merge: merge, tok: tok, max: max, temp: sp.Temperature, deep: deep}
	ch, g := f.record(c, nil)
	f.calls[len(f.calls)-1].rows = rows
	return ch, g
}

func (f *fakeMedia) GenerateVLCausalSpans(_ context.Context, ids []int, spans []decoder.ImageSpan, feat func() ([]float32, error), max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation) {
	return f.record(mediaCall{name: "causal", ids: ids, spans: spans, max: max, temp: sp.Temperature}, feat)
}

func (f *fakeMedia) GenerateVLSpans(_ context.Context, ids []int, spans []decoder.ImageSpan, feat func() ([]float32, error), max int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation) {
	return f.record(mediaCall{name: "vl", ids: ids, spans: spans, max: max, temp: sp.Temperature}, feat)
}

func TestMedia_eachKindTakesItsGenerator(t *testing.T) {
	ids := []int{1, 2, 3, 4, 5, 6}
	spans := []decoder.ImageSpan{{Pos: 1, Len: 2, Hash: 9}, {Pos: 4, Len: 1, Hash: 8}}
	rows := []float32{1, 2, 3, 4, 5, 6}
	feat := func() ([]float32, error) { return rows, nil }
	grids := [][3]int{{1, 2, 2}, {1, 1, 2}}
	for _, c := range []struct {
		md   Media
		want mediaCall
	}{
		{Media{Kind: MediaImage, Spans: spans, Features: feat}, mediaCall{name: "vl", spans: spans}},
		{Media{Kind: MediaImageCausal, Spans: spans, Features: feat}, mediaCall{name: "causal", spans: spans}},
		{Media{Kind: MediaGemma4, Spans: spans, Features: feat}, mediaCall{name: "gemma4", spans: spans}},
		{Media{Kind: MediaAudio, Spans: spans[:1], Features: feat}, mediaCall{name: "audio", pos: 1, n: 2}},
		{Media{Kind: MediaQwen, Spans: spans, Features: feat, Grids: grids, Merge: 2, ImageToken: 77}, mediaCall{name: "qwen", spans: spans, grids: grids, merge: 2, tok: 77}},
	} {
		f := &fakeMedia{hidden: 2}
		tokens, _ := generateMedia(context.Background(), f, Request{PromptIDs: ids, MaxTokens: 11, Sampling: decoder.SamplingParams{Temperature: 0.5}, Media: &c.md})
		for range tokens {
		}
		want := c.want
		want.ids, want.max, want.temp, want.rows = ids, 11, 0.5, rows
		if len(f.calls) != 1 || !reflect.DeepEqual(f.calls[0], want) {
			t.Errorf("kind %d: calls %+v, want one %+v", c.md.Kind, f.calls, want)
		}
	}
}

func TestMedia_deepStackSplitsByHiddenDim(t *testing.T) {
	// two spans of 1 and 2 rows, HiddenDim 2: 6 merged values, then two DeepStack sets of 6 each
	flat := make([]float32, 18)
	for i := range flat {
		flat[i] = float32(i)
	}
	f := &fakeMedia{hidden: 2}
	md := &Media{Kind: MediaQwen, Spans: []decoder.ImageSpan{{Pos: 0, Len: 1}, {Pos: 3, Len: 2}}, DeepSets: 2,
		Features: func() ([]float32, error) { return flat, nil }}
	generateMedia(context.Background(), f, Request{Media: md})
	if len(f.calls) != 1 {
		t.Fatalf("%d calls, want 1", len(f.calls))
	}
	c := f.calls[0]
	if !slices.Equal(c.rows, flat[:6]) || len(c.deep) != 2 || !slices.Equal(c.deep[0], flat[6:12]) || !slices.Equal(c.deep[1], flat[12:18]) {
		t.Errorf("rows %v, deep %v; want %v then %v and %v", c.rows, c.deep, flat[:6], flat[6:12], flat[12:18])
	}
}

func TestReport_carriesTheImageFields(t *testing.T) {
	r := reportOf(&decoder.Generation{ImgPrefillResident: true, ImgPrefillDecline: "no slot", DecodeResident: true})
	if !r.ImgPrefillResident || r.ImgPrefillDecline != "no slot" || !r.DecodeResident {
		t.Errorf("reportOf = %+v, want the three image fields copied", r)
	}
}

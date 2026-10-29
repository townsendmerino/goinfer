package serveapp

import (
	"context"
	"reflect"
	"testing"

	"github.com/townsendmerino/goinfer/engine"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// nilReportEngine streams ids on a stream with no report function at all, which the engine contract allows (a Stream's
// Report is then nil).
type nilReportEngine struct{ ids []int }

func (e nilReportEngine) Start(context.Context, engine.Request) *engine.Stream {
	ch := make(chan int, len(e.ids))
	for _, id := range e.ids {
		ch <- id
	}
	close(ch)
	return engine.NewStream(ch, nil, nil)
}

// TestDrive_engineWithNoReportFunc: drive reads the report through its Stream at every use, so an engine with a nil
// report function streams and finishes like the scripted seam (no logprobs, no reuse, no error) instead of panicking.
func TestDrive_engineWithNoReportFunc(t *testing.T) {
	lm := &loadedModel{tk: &tokenizer.Tokenizer{}, fakeEngine: nilReportEngine{}}
	finish, n, lps, stopHit, reused, cancelReason, err := lm.drive(context.Background(), genRequest{maxTokens: 4}, nil, nil, func(string) {})
	if finish != "stop" || n != 0 || lps != nil || stopHit != "" || reused != 0 || cancelReason != "" || err != nil {
		t.Fatalf("drive on a nil-report engine = (%q, %d, %v, %q, %d, %q, %v), want a clean stop and nothing else",
			finish, n, lps, stopHit, reused, cancelReason, err)
	}
}

// mediaEngine records the Request it is started with and streams nothing, on a stream with no report.
type mediaEngine struct{ got *engine.Request }

func (e mediaEngine) Start(_ context.Context, req engine.Request) *engine.Stream {
	*e.got = req
	ch := make(chan int)
	close(ch)
	return engine.NewStream(ch, nil, nil)
}

// TestDriveVL_mediaRequest: each visionInput shape driveVL handles reaches the engine as the Media its decoder
// generator took before the engine owned the choice (kind, spans with the one-media fallback, Qwen's grids, merge,
// image token and DeepStack sets), and an engine with no report ends the turn cleanly.
func TestDriveVL_mediaRequest(t *testing.T) {
	multi := []imageSpan{{Pos: 2, Len: 3, Hash: 7}, {Pos: 9, Len: 1, Hash: 8}}
	grids := [][3]int{{1, 2, 2}, {1, 1, 2}}
	one := visionInput{imgPos: 4, imgLen: 5, imgHash: 6, grid: [3]int{1, 3, 3}}
	with := func(vi visionInput, f func(*visionInput)) visionInput { f(&vi); return vi }
	for _, c := range []struct {
		name     string
		vi       visionInput
		kind     engine.MediaKind
		spans    []imageSpan
		qwenArgs bool
		grids    [][3]int
		deep     int
	}{
		{"audio", with(one, func(v *visionInput) { v.asr = true }), engine.MediaAudio, []imageSpan{{4, 5, 6}}, false, nil, 0},
		{"qwen3-vl deepstack", with(one, func(v *visionInput) { v.qwen, v.deepSets, v.spans, v.grids = true, 3, multi, grids }), engine.MediaQwen, multi, true, grids, 3},
		{"qwen one-media", with(one, func(v *visionInput) { v.qwen = true }), engine.MediaQwen, []imageSpan{{4, 5, 6}}, true, [][3]int{{1, 3, 3}}, 0},
		{"gemma4", with(one, func(v *visionInput) { v.gemma4, v.spans = true, multi }), engine.MediaGemma4, multi, false, nil, 0},
		{"causal", with(one, func(v *visionInput) { v.causal, v.spans = true, multi }), engine.MediaImageCausal, multi, false, nil, 0},
		{"bidirectional", with(one, func(v *visionInput) { v.spans = multi }), engine.MediaImage, multi, false, nil, 0},
	} {
		var got engine.Request
		lm := &loadedModel{tk: &tokenizer.Tokenizer{}, fakeEngine: mediaEngine{&got}, qwenMerge: 2, qwenImgTok: 151655}
		ran := false
		c.vi.features = func() ([]float32, error) { ran = true; return nil, nil }
		gr := genRequest{promptIDs: []int{1, 2, 3}, maxTokens: 4}
		gr.sp.Temperature, gr.sp.TopK = 0.7, 5
		finish, n, lps, stopHit, reused, cancelReason, err := lm.driveVL(context.Background(), gr, c.vi, nil, nil, func(string) {})
		if finish != "stop" || n != 0 || lps != nil || stopHit != "" || reused != 0 || cancelReason != "" || err != nil {
			t.Errorf("%s: driveVL on a no-report engine = (%q, %d, %v, %q, %d, %q, %v), want a clean stop", c.name, finish, n, lps, stopHit, reused, cancelReason, err)
		}
		md := got.Media
		if md == nil {
			t.Fatalf("%s: the engine got no Media", c.name)
		}
		if got.Sampling.Temperature != 0.7 || got.Sampling.TopK != 5 {
			t.Errorf("%s: sampling %+v, want the request's (temperature 0.7, top-k 5)", c.name, got.Sampling)
		}
		if md.Kind != c.kind || got.MaxTokens != 4 || len(got.PromptIDs) != 3 {
			t.Errorf("%s: kind %d, max %d, prompt %v; want kind %d, max 4, the prompt", c.name, md.Kind, got.MaxTokens, got.PromptIDs, c.kind)
		}
		if len(md.Spans) != len(c.spans) {
			t.Fatalf("%s: %d spans, want %d", c.name, len(md.Spans), len(c.spans))
		}
		for i, sp := range md.Spans {
			if w := c.spans[i]; sp.Pos != w.Pos || sp.Len != w.Len || sp.Hash != w.Hash {
				t.Errorf("%s: span %d = %+v, want %+v", c.name, i, sp, w)
			}
		}
		if md.Features == nil {
			t.Fatalf("%s: no features", c.name)
		}
		if md.Features(); !ran {
			t.Errorf("%s: Media.Features is not the input's features", c.name)
		}
		wantMerge, wantTok := 0, 0
		if c.qwenArgs {
			wantMerge, wantTok = 2, 151655
		}
		if md.Merge != wantMerge || md.ImageToken != wantTok || md.DeepSets != c.deep || (c.grids != nil || md.Grids != nil) && !reflect.DeepEqual(md.Grids, c.grids) {
			t.Errorf("%s: grids %v merge %d token %d sets %d; want %v %d %d %d", c.name, md.Grids, md.Merge, md.ImageToken, md.DeepSets, c.grids, wantMerge, wantTok, c.deep)
		}
	}
}

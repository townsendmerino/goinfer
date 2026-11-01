package engine

import (
	"context"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/cost"
)

// freshPool hands out a new session for every generation.
type freshPool struct{ m *decoder.Model }

func (p freshPool) Acquire([]int) *decoder.Session { return p.m.NewSession(64) }
func (freshPool) Checkin(*decoder.Session)         {}

// TestLocal_costReachesOnlyThePlainPaths (G-SB11b-2): a counter on the request's context fills the report on the plain session
// path, and the speculative path, which does not count, leaves it empty rather than partly counted; a media request's generator
// sees no counter.
func TestLocal_costReachesOnlyThePlainPaths(t *testing.T) {
	m, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Quant: "f32"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	run := func(d Local) cost.Report {
		c := cost.New()
		st := d.Start(cost.WithCounter(context.Background(), c), Request{PromptIDs: []int{1, 2, 3, 4, 5, 6}, MaxTokens: 4,
			Sampling: decoder.SamplingParams{Temperature: 0}})
		for range st.Tokens() {
		}
		if r := st.Report(); r != nil && r.Err != nil {
			t.Fatal(r.Err)
		}
		st.Close()
		return c.Report()
	}
	if r := run(Local{Model: m, Pool: freshPool{m}}); r.Prefill == nil || r.Decode == nil {
		t.Errorf("the plain session path counted %+v, want a prefill and a decode phase", r)
	}
	if r := run(Local{Model: m, Pool: freshPool{m}, Speculative: true}); r.Prefill != nil || r.Decode != nil {
		t.Errorf("the speculative path counted %+v, want an empty report", r)
	}

	f := &fakeMedia{hidden: 2}
	md := &Media{Kind: MediaGemma4, Spans: []decoder.ImageSpan{{Pos: 0, Len: 1}}, Features: func() ([]float32, error) { return nil, nil }}
	st := Local{media: f}.Start(cost.WithCounter(context.Background(), cost.New()), Request{PromptIDs: []int{1}, MaxTokens: 2, Media: md})
	for range st.Tokens() {
	}
	st.Close()
	if !f.sawCtx || f.counter != nil {
		t.Errorf("the media generator saw a context %v with counter %v, want one with none", f.sawCtx, f.counter)
	}
}

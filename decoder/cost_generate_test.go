package decoder

import (
	"context"
	"fmt"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/internal/cost"
)

// generateCounted runs gen on llama-tiny with c on the context (nil: off) and returns the emitted tokens.
func generateCounted(t *testing.T, gen func(ctx context.Context) (<-chan int, *Generation), c *cost.Counter) []int {
	t.Helper()
	stream, g := gen(cost.WithCounter(context.Background(), c))
	var out []int
	for id := range stream {
		out = append(out, id)
	}
	if err := g.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

// costGenerateErrors grades a whole generation's report (G-SB11b-3): the prefill covers the prompt, reading one embedding row
// per prompt token; the decode covers one forward per emitted token, each running every operation of the program once, plus
// its embedding, final norm and head.
func costGenerateErrors(m *Model, r cost.Report, promptLen, emitted int) []string {
	var errs []string
	p, d := r.Prefill, r.Decode
	if p == nil {
		return []string{"no prefill phase"}
	}
	if p.Tokens != int64(promptLen) {
		errs = append(errs, fmt.Sprintf("prefill tokens %d, want the prompt's %d", p.Tokens, promptLen))
	}
	embedRow := wmBytes(&m.w.Embed) / int64(m.w.Embed.Rows())
	if got, want := p.Ops[costEmbed].BytesRead, int64(promptLen)*embedRow; got != want {
		errs = append(errs, fmt.Sprintf("prefill embedding read %d bytes, want %d (%d rows)", got, want, promptLen))
	}
	if emitted == 0 {
		if d != nil {
			errs = append(errs, fmt.Sprintf("no token emitted, but a decode phase %+v", d))
		}
		return errs
	}
	if d == nil {
		return append(errs, "no decode phase")
	}
	n := int64(emitted)
	if d.Tokens != n {
		errs = append(errs, fmt.Sprintf("decode tokens %d, want one per emitted token, %d", d.Tokens, n))
	}
	want := map[string]int64{costEmbed: n, costFinalNorm: n, costLMHead: n}
	for _, ops := range m.program().Layers {
		for _, o := range ops {
			want[opCostNames[o.Kind]] += n
		}
	}
	for name, c := range want {
		if got := d.Ops[name].Calls; got != c {
			errs = append(errs, fmt.Sprintf("decode %s: %d calls, want %d", name, got, c))
		}
	}
	for name := range d.Ops {
		if want[name] == 0 {
			errs = append(errs, fmt.Sprintf("decode %s counted, but the program has none", name))
		}
	}
	return errs
}

// TestCost_generateOnTheWire (G-SB11b-3): Model.Generate and Session.Generate with a counter on the context fill a report
// whose prefill covers the prompt and whose decode covers each emitted token's forward, and they emit the same tokens as
// with no counter. With no counter on the context, no cache is left counting.
func TestCost_generateOnTheWire(t *testing.T) {
	m := loadCostFixture(t)
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8, 9}
	sp := SamplingParams{Temperature: 0}
	for _, tc := range []struct {
		name string
		gen  func(ctx context.Context) (<-chan int, *Generation)
	}{
		{"Model.Generate", func(ctx context.Context) (<-chan int, *Generation) { return m.Generate(ctx, prompt, 6, sp) }},
		{"Session.Generate", func(ctx context.Context) (<-chan int, *Generation) {
			return m.NewSession(64).Generate(ctx, prompt, 6, sp)
		}},
	} {
		off := generateCounted(t, tc.gen, nil)
		c := cost.New()
		on := generateCounted(t, tc.gen, c)
		if !slices.Equal(on, off) {
			t.Errorf("%s: tokens %v with the counter on, %v off", tc.name, on, off)
		}
		for _, e := range costGenerateErrors(m, c.Report(), len(prompt), len(on)) {
			t.Errorf("%s: %s", tc.name, e)
		}
	}
	// A session's cache outlives its generation: after a counted turn it must not keep counting into that turn's counter.
	s := m.NewSession(64)
	c := cost.New()
	generateCounted(t, func(ctx context.Context) (<-chan int, *Generation) { return s.Generate(ctx, prompt, 3, sp) }, c)
	before := c.Report()
	generateCounted(t, func(ctx context.Context) (<-chan int, *Generation) { return s.Generate(ctx, prompt, 3, sp) }, nil)
	tokens := func(p *cost.PhaseReport) int64 {
		if p == nil {
			return 0
		}
		return p.Tokens
	}
	if after := c.Report(); tokens(after.Decode) != tokens(before.Decode) || tokens(after.Prefill) != tokens(before.Prefill) {
		t.Errorf("an uncounted turn on the same session counted into the earlier turn's counter: %+v then %+v", before, after)
	}
}

package decoder

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"sort"
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
)

// policyStreamsPath is SB5's census (docs/tasks/task-subsystems-2026-10.md §8 SB5, gate SB5-P1): the token stream each
// engine policy produces on committed tiny fixtures. -update rewrites it; otherwise every stream must reproduce exactly.
const policyStreamsPath = "../testdata/sb5/policy_streams.json"

// policyCase is one recorded stream: the tokens, the speculative counters when the policy speculates, the prefill reuse
// when it runs in a session, and whether the stream equals the plain-decoding stream it is held to (Plain names that
// case; empty for a policy that is not exact by stream, such as sampled speculation).
type policyCase struct {
	Tokens        []int      `json:"tokens"`
	Spec          *SpecStats `json:"spec,omitempty"`
	PrefillReused int        `json:"prefill_reused,omitempty"`
	Plain         string     `json:"plain,omitempty"`
}

// censusPrompts are byte-level prompts for the 256-token fixtures: a repeating phrase (n-gram drafts land), plain
// text, and a short one.
var censusPrompts = [][]int{
	bytesIDs("the cat sat on the mat. the cat sat on the mat. the cat"),
	bytesIDs("Speculative decoding verifies a draft in one forward."),
	bytesIDs("{\"a\":"),
}

func bytesIDs(s string) []int {
	out := make([]int, len(s))
	for i := range len(s) {
		out[i] = int(s[i])
	}
	return out
}

// byteVocab is a byte-level vocabulary: token i is the byte i.
func byteVocab(n int) [][]byte {
	v := make([][]byte, n)
	for i := range v {
		v[i] = []byte{byte(i)}
	}
	return v
}

const censusSchema = `{"type":"object","properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},"required":["location","unit"],"additionalProperties":false}`

// TestPolicyStreams_census is SB5-P1: every engine policy's token stream on llama-tiny (and llama-attnfa-tiny as a
// draft model), recorded and held. An exact policy's stream must also equal the plain-decoding stream it names.
func TestPolicyStreams_census(t *testing.T) {
	got := policyCensus(t)
	for name, c := range got {
		if c.Plain == "" {
			continue
		}
		ref, ok := got[c.Plain]
		if !ok {
			t.Fatalf("%s is held to %s, which the census does not have", name, c.Plain)
		}
		if !slices.Equal(c.Tokens, ref.Tokens) {
			t.Errorf("%s is not exact: its stream differs from %s\n got %v\n ref %v", name, c.Plain, c.Tokens, ref.Tokens)
		}
	}
	b, err := json.MarshalIndent(got, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	b = append(b, '\n')
	if *updateMatrix {
		if err := os.MkdirAll("../testdata/sb5", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(policyStreamsPath, b, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d cases to %s", len(got), policyStreamsPath)
		return
	}
	want, err := os.ReadFile(policyStreamsPath)
	if err != nil {
		t.Fatalf("read %s: %v (run with -update to create it)", policyStreamsPath, err)
	}
	if string(want) != string(b) {
		var w map[string]policyCase
		if err := json.Unmarshal(want, &w); err != nil {
			t.Fatal(err)
		}
		var names []string
		for n := range got {
			names = append(names, n)
		}
		for n := range w {
			if _, ok := got[n]; !ok {
				names = append(names, n)
			}
		}
		sort.Strings(names)
		for _, n := range names {
			a, _ := json.Marshal(w[n])
			g, _ := json.Marshal(got[n])
			if string(a) != string(g) {
				t.Errorf("%s changed:\n want %s\n got  %s", n, a, g)
			}
		}
	}
	t.Logf("%d policy streams", len(got))
}

// policyCensus runs every case.
func policyCensus(t *testing.T) map[string]policyCase {
	t.Helper()
	m, err := Load("../testdata/llama-tiny", Options{})
	if err != nil {
		t.Fatalf("llama-tiny: %v", err)
	}
	draft, err := Load("../testdata/llama-attnfa-tiny", Options{})
	if err != nil {
		t.Fatalf("llama-attnfa-tiny: %v", err)
	}
	self, err := Load("../testdata/llama-tiny", Options{})
	if err != nil {
		t.Fatalf("llama-tiny as a draft: %v", err)
	}
	ctx := context.Background()
	const n = 24
	greedy := SamplingParams{Temperature: 0}
	out := map[string]policyCase{}
	run := func(name, plain string, ch <-chan int, g *Generation, err error) []int {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		toks := collectTokens(ch)
		if g.Err() != nil {
			t.Fatalf("%s: stream: %v", name, g.Err())
		}
		c := policyCase{Tokens: toks, Plain: plain, PrefillReused: g.PrefillReused}
		if g.Spec != nil {
			s := *g.Spec
			c.Spec = &s
		}
		if c.Tokens == nil {
			c.Tokens = []int{}
		}
		out[name] = c
		return toks
	}
	plain := func(name string, p []int, sp SamplingParams) []int {
		ch, g := m.Generate(ctx, p, n, sp)
		return run(name, "", ch, g, nil)
	}

	for pi, p := range censusPrompts {
		ref := fmt.Sprintf("plain/p%d", pi)
		plain(ref, p, greedy)
		for _, K := range []int{1, 4, 8} {
			ch, g, err := m.GenerateNgramSpeculative(ctx, p, n, &NgramDrafter{}, K, greedy)
			run(fmt.Sprintf("ngram-K%d/p%d", K, pi), ref, ch, g, err)
		}
		ch, g, err := m.GenerateNgramSpeculativeAdaptive(ctx, p, n, &NgramDrafter{}, &AdaptiveDepth{MaxDraft: 8, Theta: 0.5}, greedy)
		run(fmt.Sprintf("ngram-adaptive/p%d", pi), ref, ch, g, err)
		ch, g, err = m.GenerateSpeculative(ctx, p, n, draft, 4, greedy)
		run(fmt.Sprintf("draft-attnfa-K4/p%d", pi), ref, ch, g, err)
		ch, g, err = m.GenerateSpeculative(ctx, p, n, self, 4, greedy)
		run(fmt.Sprintf("draft-self-K4/p%d", pi), ref, ch, g, err)

		sampled := SamplingParams{Temperature: 0.8, TopK: 40, Seed: 7}
		plain(fmt.Sprintf("sampled-plain/p%d", pi), p, sampled)
		ch, g, err = m.GenerateNgramSpeculative(ctx, p, n, &NgramDrafter{}, 4, sampled)
		run(fmt.Sprintf("sampled-ngram-K4/p%d", pi), "", ch, g, err)
	}

	// Grammar: constrained greedy as the reference, then grammar speculation with the grammar drafter.
	vocab := byteVocab(m.w.arch.VocabSize)
	newMask := func() *constrain.Masker {
		gr, err := constrain.JSONSchema([]byte(censusSchema))
		if err != nil {
			t.Fatalf("JSONSchema: %v", err)
		}
		return constrain.NewMasker(gr, vocab, m.eosIDs).StopWhenComplete()
	}
	gp := bytesIDs("Weather for Paris as JSON: ")
	refSP := greedy
	refSP.LogitProcessor = newMask().Process
	ch, g := m.Generate(ctx, gp, 64, refSP)
	run("grammar-plain", "", ch, g, nil)
	for _, K := range []int{1, 8} {
		mk := newMask()
		gd := &GrammarDrafter{Mask: mk, Encode: func(s string) []int { return bytesIDs(s) }}
		ch, g, err := m.GenerateGrammarSpeculative(ctx, gp, 64, mk, gd, K, greedy)
		run(fmt.Sprintf("grammar-spec-K%d", K), "grammar-plain", ch, g, err)
	}

	// Sessions: a resumed turn, a rewound one and a copied prefix, each held to cold decoding of the same prompt.
	s := m.NewSession(512)
	p0 := censusPrompts[0]
	ch, g = s.Generate(ctx, p0, n, greedy)
	t1 := run("session/turn1", "plain/p0", ch, g, nil)
	p1 := append(append(slices.Clone(p0), t1...), bytesIDs(" and then")...)
	plain("plain/session-turn2", p1, greedy)
	ch, g = s.Generate(ctx, p1, n, greedy)
	run("session/turn2-resume", "plain/session-turn2", ch, g, nil)
	p2 := append(slices.Clone(p0[:20]), bytesIDs(" dog ran off")...)
	plain("plain/session-turn3", p2, greedy)
	ch, g = s.Generate(ctx, p2, n, greedy)
	run("session/turn3-rewind", "plain/session-turn3", ch, g, nil)

	c := m.NewSession(512)
	if err := c.CopyPrefixFrom(s, 16); err != nil {
		t.Fatalf("CopyPrefixFrom: %v", err)
	}
	ch, g = c.Generate(ctx, p2, n, greedy)
	run("session/copied-prefix", "plain/session-turn3", ch, g, nil)

	ss := m.NewSession(512)
	ch, g, err = ss.GenerateNgramSpeculative(ctx, p0, n, &NgramDrafter{}, 4, greedy)
	st1 := run("session-ngram/turn1", "plain/p0", ch, g, err)
	p3 := append(append(slices.Clone(p0), st1...), bytesIDs(" and then")...)
	ch, g, err = ss.GenerateNgramSpeculative(ctx, p3, n, &NgramDrafter{}, 4, greedy)
	run("session-ngram/turn2-resume", "plain/session-turn2", ch, g, err)
	return out
}

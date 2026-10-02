//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGreedyChain_generateMatchesFullLogits is C-B01 through the decoder: greedy Generate on Metal with the chain
// (ResidentGreedyChain, the default where it is exact) against the same request with the chain off (greedyChainOff:
// the full-logits ForwardEmbPipe path production ran before). One generation of 90 tokens, past the chain's pool drain;
// then two turns, the second's prompt the first's prompt and reply plus three tokens, so the resident reuses the
// prefix and the forward the chain queued past turn one's last token must not leak into turn two. (A Session is not
// the multi-turn path on a resident: a plain session decodes on the CPU path, session.go.) The chain must have served
// every turn's tokens, or the comparison is of a path with itself.
func TestGreedyChain_generateMatchesFullLogits(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := writeMC3FixtureTied(t, 4096, true)
	prompt := make([]int, 24)
	for i := range prompt {
		prompt[i] = (i*613 + 101) % mfVocab
	}
	type out struct {
		one, turn1, turn2 []int
		served, reused    int
	}
	run := func(off bool) out {
		greedyChainOff = off
		defer func() { greedyChainOff = false }()
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024, ResidentKVSlots: 1})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		defer m.Close()
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("no metal resident: %s", m.ResidentDecline())
		}
		ctx := context.Background()
		var o out
		gen := func(p []int, n int) []int {
			ch, g := m.Generate(ctx, p, n, decoder.SamplingParams{})
			var ids []int
			for id := range ch {
				ids = append(ids, id)
			}
			if err := g.Err(); err != nil {
				t.Fatalf("generate: %v", err)
			}
			o.reused = g.PrefillReused
			return ids
		}
		o.one = gen(prompt, 90)
		o.turn1 = gen(prompt, 30)
		ext := append(append(append([]int(nil), prompt...), o.turn1...), 777, 778, 779)
		o.turn2 = gen(ext, 30)
		o.served = a.r.chainServed
		return o
	}
	ref, got := run(true), run(false)
	t.Logf("chain served %d tokens (off: %d); one %d tokens, turns %d + %d; turn two reused %d of %d prompt tokens (off: %d)",
		got.served, ref.served, len(got.one), len(got.turn1), len(got.turn2), got.reused, len(prompt)+30+3, ref.reused)
	if ref.served != 0 {
		t.Fatalf("greedyChainOff still served %d tokens", ref.served)
	}
	if want := 90 + 30 + 30; got.served < want-3 { // each generation's first token comes from its prompt's logits
		t.Fatalf("the chain served %d tokens of the three generations' %d: it did not serve them all", got.served, want)
	}
	if got.reused == 0 || got.reused != ref.reused {
		t.Fatalf("turn two reused %d prompt tokens with the chain, %d without: the test needs the resident's prefix reuse on both", got.reused, ref.reused)
	}
	if !slices.Equal(got.one, ref.one) {
		t.Fatalf("one generation: the chain's tokens %v differ from the full-logits path's %v", got.one, ref.one)
	}
	if !slices.Equal(got.turn1, ref.turn1) || !slices.Equal(got.turn2, ref.turn2) {
		t.Fatalf("two turns: the chain's turns %v / %v differ from the full-logits path's %v / %v", got.turn1, got.turn2, ref.turn1, ref.turn2)
	}
}

//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillLast_tailContinuationMatchesCold (decoder.PrefillTailExact): a PrefillLast of 1, 2, 7 or 8 tokens that
// continues a prefix the batched pass prefilled computes the cold whole-prompt pass's last row bit for bit, so a reused
// prompt can hand its short suffix to the pass. The decode continuation the decoder used to run for such a suffix
// differs in every logit, which is the warm-against-cold divergence this exists to close.
func TestPrefillLast_tailContinuationMatchesCold(t *testing.T) {
	m, err := decoder.Load(mc3ChainFixture, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024})
	if err != nil {
		t.Skipf("load %s: %v", mc3ChainFixture, err)
	}
	defer m.Close()
	a := m.ResidentForwardForTest().(*metalResident)
	if !a.PrefillTailExact() {
		t.Fatal("the dense fixture does not claim PrefillTailExact: this gate would test nothing")
	}
	const n = 100
	embs := getEmbs(a.r, mc3ChainPrompt(2, n))
	cold, err := a.PrefillLast(context.Background(), embs, 0)
	if err != nil {
		t.Fatalf("cold pass: %v", err)
	}
	cold = append([]float32(nil), cold...)
	differ := func(x []float32) int {
		d := 0
		for i := range cold {
			if math.Float32bits(cold[i]) != math.Float32bits(x[i]) {
				d++
			}
		}
		return d
	}
	for _, tail := range []int{1, 2, 7, 8} {
		if _, err := a.PrefillLast(context.Background(), embs[:n-tail], 0); err != nil {
			t.Fatal(err)
		}
		warm, err := a.PrefillLast(context.Background(), embs[n-tail:], n-tail)
		if err != nil {
			t.Fatalf("tail %d: %v", tail, err)
		}
		if d := differ(warm); d != 0 {
			t.Errorf("tail %d: a pass continuing the cached prefix differs from the cold pass in %d of %d logits", tail, d, len(cold))
		}
	}
	// The control: the same 1-token tail through decode, as the decoder ran it before PrefillTailExact.
	if _, err := a.PrefillLast(context.Background(), embs[:n-1], 0); err != nil {
		t.Fatal(err)
	}
	if d := differ(a.r.ForwardEmb(embs[n-1], n-1)); d == 0 {
		t.Fatal("decode of the last position equals the cold pass: the pass is no longer a different lane, and this gate tests nothing")
	}
}

// TestGenerate_warmRepeatMatchesCold: the same prompt generated twice on one model emits the same tokens, the second time
// reusing every cached position but the last. Before decoder.PrefillTailExact, a prompt the cold run had prefilled with
// the batched pass ran its last position through decode on the repeat and diverged some tens of tokens later (on the
// 2-slot fixture, token 42 of 160); TestMC3Chain_newcomerJoinsAndBothMatchAlone met it as an "MC3 identity defect"
// because its alone runs were cold and its joint runs warm. Single- and 2-slot models, at 12 tokens (below the 16-token
// floor: the exact lane both ways), 40 (the pass on one slot, the step on two) and 100 (the pass), greedy.
func TestGenerate_warmRepeatMatchesCold(t *testing.T) {
	for _, slots := range []int{1, 2} {
		for _, n := range []int{12, 40, 100} {
			t.Run(fmt.Sprintf("slots=%d/n=%d", slots, n), func(t *testing.T) {
				prompt := mc3ChainPrompt(7, n)
				run := func(m *decoder.Model, maxTok int) ([]int, int) {
					ch, g := m.Generate(context.Background(), prompt, maxTok, decoder.SamplingParams{})
					var ids []int
					for id := range ch {
						ids = append(ids, id)
					}
					if err := g.Err(); err != nil {
						t.Fatalf("generate: %v", err)
					}
					return ids, g.PrefillReused
				}
				// cold: a fresh model. warm: another fresh model whose slot was primed with the prompt and 4 tokens, so
				// the repeat reuses n-1 positions and discards 5 (the slot picker keeps a reuse that keeps more than it
				// discards; after a long reply a short prompt rightly goes cold to another slot instead).
				mc := mc3ChainModel(t, slots)
				cold, r0 := run(mc, 96)
				mc.Close()
				mw := mc3ChainModel(t, slots)
				defer mw.Close()
				run(mw, 4)
				warm, r1 := run(mw, 96)
				if r0 != 0 || r1 != n-1 {
					t.Fatalf("reused %d (cold) and %d (warm) prompt positions, want 0 and %d: the second is not the warm repeat this tests", r0, r1, n-1)
				}
				if !slices.Equal(cold, warm) {
					i := 0
					for i < min(len(cold), len(warm)) && cold[i] == warm[i] {
						i++
					}
					t.Fatalf("the warm repeat differs from the cold run at token %d of %d", i, len(cold))
				}
			})
		}
	}
}

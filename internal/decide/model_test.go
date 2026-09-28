package decide

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestDecide_realModel runs Route A end to end on a real Qwen checkpoint (CPU), under both templates: every bare
// verbalizer is one token in the Qwen tokenizer, the distributions are proper, and the readout moves the right way on
// a pair that differs only in the answer. The arithmetic against transformers' own restricted softmax on identical
// bytes is the D0 fixture's job (docs/prompts/nobara-decisions-d0-fixture-2026-09.md), not this test's.
func TestDecide_realModel(t *testing.T) {
	const path = "../../testdata/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no local checkpoint at %s (an untracked symlink): %v", path, err)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ctx := context.Background()
	refund := func(asks bool) Request {
		state := "The customer wrote: thanks, everything arrived and works great!"
		if asks {
			state = "The customer wrote: please refund my $20, the item never arrived."
		}
		return Request{Kind: KindNoul, State: state, Question: "Is the customer asking for a refund?", Options: noulOptions}
	}
	choice := Request{Kind: KindChoice, State: "The server returned HTTP 500 on every request since the deploy.",
		Question: "Which team should handle this?", Options: []string{"billing", "engineering", "sales"}}
	for _, tmpl := range []string{TemplateBare, TemplateChat} {
		t.Run(tmpl, func(t *testing.T) {
			d, err := New(NewPlainTokenizer(tk), ModelPrefill(m), Options{Template: tmpl})
			if err != nil {
				t.Fatal(err)
			}
			for _, v := range []string{"false", "true", "0", "5", "A", "P"} {
				if _, ok := d.LabelID(v); !ok {
					t.Errorf("label %q unresolved", v)
				}
			}
			// Mechanics, not model quality: proper distributions, and the refund question's P(true) higher for the
			// customer who asks for one than for the one who does not. How good the answers are is D6a's to measure.
			var pTrue [2]float64
			for i, asks := range []bool{true, false} {
				r, err := d.Decide(ctx, refund(asks))
				if err != nil {
					t.Fatal(err)
				}
				checkProper(t, r)
				pTrue[i] = r.Distribution[1]
				t.Logf("refund asks=%v → %s %.3f %v (%d tokens, %s)", asks, r.Decision, r.Confidence, r.Distribution, r.PromptTokens, r.Latency)
			}
			if !(pTrue[0] > pTrue[1]) {
				t.Errorf("P(true) %.3f for a refund request is not above %.3f for none", pTrue[0], pTrue[1])
			}
			r, err := d.Decide(ctx, choice)
			if err != nil {
				t.Fatal(err)
			}
			checkProper(t, r)
			t.Logf("choice → %s %.3f %v", r.Decision, r.Confidence, r.Distribution)
		})
	}
}

func checkProper(t *testing.T, r Result) {
	t.Helper()
	var sum float64
	for _, p := range r.Distribution {
		if p < 0 || p > 1 || math.IsNaN(p) {
			t.Errorf("probability %v", p)
		}
		sum += p
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Errorf("distribution sums to %v", sum)
	}
}

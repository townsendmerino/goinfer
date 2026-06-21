package decoder

import (
	"context"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGrammarRouterSpec gates 03 increment 1 (priority router): fusing a grammar
// source (forces the scaffolding) with an n-gram source (copies free values that
// echo the prompt) under one masked verify. Both must stay token-identical to plain
// constrained Generate (the verify decides, not the drafter), and the router should
// commit at least as many tokens/round as grammar alone — the n-gram picks up the
// free string value ("Paris") the grammar leaves open.
func TestGrammarRouterSpec(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	tk, err := tokenizer.LoadGGUF(benchGGUFPath())
	if err != nil {
		t.Skipf("no tokenizer (%v)", err)
	}
	vocab := m.w.arch.VocabSize
	vocabBytes := constrain.TokenBytes(vocab, tk.TokenText)
	encode := func(s string) []int { ids, _ := tk.Encode(s, false); return ids }
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}
	const n = 64

	schema := `{"type":"object","properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},"required":["location","unit"],"additionalProperties":false}`
	// Agent-loop / repeated-tool-call case (the regime the 03 doc cites for fusion):
	// the prompt contains the exact JSON to reproduce, so the n-gram source copies
	// the whole object — including the free value the grammar leaves open — in long
	// runs, while the grammar validates. This is where grammar+n-gram compound.
	priorJSON := "{\n  \"location\": \"Paris\",\n  \"unit\": \"celsius\"\n}"
	prompt, _ := tk.Encode("<|im_start|>user\nRepeat this JSON exactly:\n"+priorJSON+"<|im_end|>\n<|im_start|>assistant\n", true)
	newMask := func() *constrain.Masker {
		g, gerr := constrain.JSONSchema([]byte(schema))
		if gerr != nil {
			t.Fatalf("JSONSchema: %v", gerr)
		}
		return constrain.NewMasker(g, vocabBytes, m.eosIDs).StopWhenComplete()
	}

	// Reference: plain constrained Generate.
	refSP := greedy
	refSP.LogitProcessor = newMask().Process
	refCh, _ := m.Generate(ctx, prompt, n, refSP)
	ref := collectTokens(refCh)

	run := func(d Drafter) ([]int, *SpecStats) {
		mask := newMask()
		// rebind any GrammarDrafter source to this run's mask
		if rd, ok := d.(*RouterDrafter); ok {
			for _, s := range rd.Sources {
				if gd, ok := s.(*GrammarDrafter); ok {
					gd.Mask = mask
				}
			}
		} else if gd, ok := d.(*GrammarDrafter); ok {
			gd.Mask = mask
		}
		ch, g, err := m.GenerateGrammarSpeculative(ctx, prompt, n, mask, d, 8, greedy)
		if err != nil {
			t.Fatalf("spec: %v", err)
		}
		return collectTokens(ch), g.Spec
	}

	grOnly, gs := run(&GrammarDrafter{Encode: encode})
	router, rs := run(&RouterDrafter{Sources: []Drafter{&GrammarDrafter{Encode: encode}, &NgramDrafter{}}})

	if !slices.Equal(grOnly, ref) {
		t.Fatalf("grammar-only != constrained greedy")
	}
	if !slices.Equal(router, ref) {
		t.Fatalf("router != constrained greedy (fusion broke losslessness)")
	}
	t.Logf("grammar-only: %.2f tok/round (acc %.3f) | router(grammar+ngram): %.2f tok/round (acc %.3f)",
		gs.TokensPerRound(), gs.AcceptanceRate(), rs.TokensPerRound(), rs.AcceptanceRate())
	if rs.TokensPerRound() < gs.TokensPerRound()-1e-6 {
		t.Errorf("router committed fewer tok/round (%.2f) than grammar alone (%.2f) — fusion should not regress",
			rs.TokensPerRound(), gs.TokensPerRound())
	}
}

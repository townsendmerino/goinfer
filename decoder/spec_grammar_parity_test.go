package decoder

import (
	"context"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGrammarSpecParity is the 01 inc-3 gate: grammar-fused speculative decode must
// be token-identical to plain constrained Generate (greedy, same grammar) — the
// forced byte-runs are drafted for free but the masked verify reproduces exactly what
// the constrained sampler would emit. Exercises the masked verify + per-block grammar
// rollback + the forced-bytes drafter end to end.
func TestGrammarSpecParity(t *testing.T) {
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

	schema := `{"type":"object","properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},"required":["location","unit"],"additionalProperties":false}`
	// chatml + a primed "{" so the instruct model commits to emitting the object
	// (rather than greedily looping leading whitespace under the permissive mask).
	prompt, err := tk.Encode("<|im_start|>user\nWeather for Paris in celsius as JSON matching the schema.<|im_end|>\n<|im_start|>assistant\n", true)
	if err != nil {
		t.Fatalf("encode prompt: %v", err)
	}
	const n = 64
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}

	newMask := func() *constrain.Masker {
		g, gerr := constrain.JSONSchema([]byte(schema))
		if gerr != nil {
			t.Fatalf("JSONSchema: %v", gerr)
		}
		return constrain.NewMasker(g, vocabBytes, m.eosIDs).StopWhenComplete()
	}

	// Reference: plain constrained Generate (masker as LogitProcessor).
	refSP := greedy
	refSP.LogitProcessor = newMask().Process
	refCh, _ := m.Generate(ctx, prompt, n, refSP)
	ref := collectTokens(refCh)

	// Grammar-fused speculative.
	ch, g, err := m.GenerateGrammarSpeculative(ctx, prompt, n, newMask(), encode, 8, greedy)
	if err != nil {
		t.Fatalf("GenerateGrammarSpeculative: %v", err)
	}
	got := collectTokens(ch)
	if g.Err() != nil {
		t.Fatalf("spec stream err: %v", g.Err())
	}

	if !slices.Equal(got, ref) {
		t.Fatalf("grammar-fused spec != constrained greedy\n got %v\n ref %v\n got=%q\n ref=%q",
			got, ref, decodeAll(tk, got), decodeAll(tk, ref))
	}
	out, _ := tk.Decode(got)
	t.Logf("grammar-spec parity OK: %d tok, acceptance %.3f, %.2f tok/round, out=%q", len(got), g.Spec.AcceptanceRate(), g.Spec.TokensPerRound(), out)
}

func decodeAll(tk *tokenizer.Tokenizer, ids []int) string {
	s, _ := tk.Decode(ids)
	return s
}

package decoder

import (
	"context"
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGrammarSpecHarness characterizes the 01 win across structured outputs of
// varying density: how many tokens/verify does grammar-fused speculation commit when
// the schema is mostly free strings vs mostly fixed keys / enum literals? The mask
// guarantees schema-conforming output regardless of model quality, so the forced-run
// acceptance is a property of the schema × tokenizer, measured here. Logs; run -v.
// This is the data that decides whether the cmd/serve wiring is worth it.
func TestGrammarSpecHarness(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
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
	const n = 96

	cases := []struct{ name, schema string }{
		{"free-string", `{"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false}`},
		{"mixed", `{"type":"object","properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},"required":["location","unit"],"additionalProperties":false}`},
		{"enum-heavy", `{"type":"object","properties":{"status":{"enum":["approved","rejected"]},"priority":{"enum":["high","low"]},"category":{"enum":["bug","feature"]}},"required":["status","priority","category"],"additionalProperties":false}`},
		{"fixed-keys", `{"type":"object","properties":{"first_name":{"enum":["Ada"]},"last_name":{"enum":["Lovelace"]},"role":{"enum":["engineer"]}},"required":["first_name","last_name","role"],"additionalProperties":false}`},
	}

	t.Logf("%-12s %6s %8s %9s  %s", "schema", "tok", "accept", "tok/round", "output")
	for _, c := range cases {
		newMask := func() *constrain.Masker {
			g, gerr := constrain.JSONSchema([]byte(c.schema))
			if gerr != nil {
				t.Fatalf("%s: JSONSchema: %v", c.name, gerr)
			}
			return constrain.NewMasker(g, vocabBytes, m.eosIDs).StopWhenComplete()
		}
		prompt, _ := tk.Encode("<|im_start|>user\nEmit a JSON object matching the schema.<|im_end|>\n<|im_start|>assistant\n", true)

		mask := newMask()
		gd := &GrammarDrafter{Mask: mask, Encode: encode}
		ch, g, err := m.GenerateGrammarSpeculative(ctx, prompt, n, mask, gd, 8, greedy)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		got := collectTokens(ch)
		if g.Err() != nil {
			t.Fatalf("%s: stream err %v", c.name, g.Err())
		}
		out, _ := tk.Decode(got)
		t.Logf("%-12s %6d %8.3f %9.2f  %q", c.name, len(got), g.Spec.AcceptanceRate(), g.Spec.TokensPerRound(), out)
	}
}

package decoder

import (
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGrammarForcedFraction is the 01 increment-2 go/no-go: with a REAL BPE
// tokenizer (not the byte vocab), what fraction of a structured/tool-call output is
// grammar-FORCED (positions where exactly one token is legal)? That fraction is the
// ceiling on what a zero-cost grammar drafter (Masker.ForcedRun) covers for free. If
// it's material, the masked-verify integration is worth building; if it's tiny (the
// optional-whitespace caveat biting), 01 is marginal on these grammars. Logs; run -v.
func TestGrammarForcedFraction(t *testing.T) {
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

	cases := []struct {
		name, schema, output string
	}{
		{
			"weather",
			`{"type":"object","properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},"required":["location","unit"],"additionalProperties":false}`,
			`{"location":"Paris","unit":"celsius"}`,
		},
		{
			"record",
			`{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"active":{"type":"boolean"}},"required":["name","age","active"],"additionalProperties":false}`,
			`{"name":"Alice","age":30,"active":true}`,
		},
		{
			"enum-heavy",
			`{"type":"object","properties":{"status":{"enum":["approved"]},"priority":{"enum":["high"]}},"required":["status","priority"],"additionalProperties":false}`,
			`{"status":"approved","priority":"high"}`,
		},
	}

	// byteForced marks each output byte as forced (exactly one legal next byte) and
	// returns the per-byte forced mask — the BYTE-level structural-forcing ceiling.
	byteForced := func(schema, output string) []bool {
		g, err := constrain.JSONSchema([]byte(schema))
		if err != nil {
			t.Fatalf("JSONSchema: %v", err)
		}
		out := make([]bool, len(output))
		var probe [1]byte
		for i := 0; i < len(output); i++ {
			legal := 0
			for b := 0; b < 256 && legal < 2; b++ {
				probe[0] = byte(b)
				if g.TryBytes(probe[:]) {
					legal++
				}
			}
			out[i] = legal == 1
			g.Commit([]byte{output[i]})
		}
		return out
	}

	t.Logf("%-10s | TOKEN-forced (ForcedRun) | BYTE-forced (ceiling)", "schema")
	for _, c := range cases {
		// (a) token-level: what ForcedRun actually proposes (strict "one legal token").
		g, err := constrain.JSONSchema([]byte(c.schema))
		if err != nil {
			t.Fatalf("%s: JSONSchema: %v", c.name, err)
		}
		mask := constrain.NewMasker(g, vocabBytes, m.eosIDs)
		ids, err := tk.Encode(c.output, false)
		if err != nil {
			t.Fatalf("%s: encode: %v", c.name, err)
		}
		logits := make([]float32, vocab)
		tokForced := 0
		for i := range ids {
			if run := mask.ForcedRun(1); len(run) == 1 {
				tokForced++
			}
			mask.Process(ids[:i+1], logits)
		}

		// (b) byte-level ceiling: fraction of output bytes the grammar determines.
		bm := byteForced(c.schema, c.output)
		bf := 0
		var marked []byte
		for i, f := range bm {
			if f {
				bf++
				marked = append(marked, c.output[i])
			} else {
				marked = append(marked, '·')
			}
		}
		_ = bm
		t.Logf("%-10s | %2d/%2d tok = %4.1f%% | %2d/%2d bytes = %4.1f%%  %q",
			c.name, tokForced, len(ids), 100*float64(tokForced)/float64(len(ids)),
			bf, len(c.output), 100*float64(bf)/float64(len(c.output)), string(marked))
	}
}

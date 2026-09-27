package decoder

import (
	"context"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestGrammarSpec_sessionTwoTurnsMatchPlain: on a Session, a grammar-fused speculative turn that ends by reaching
// max_tokens leaves the session holding exactly what a plain constrained turn leaves (prompt + every emitted token),
// and the next turn's reply is the same. Serve runs grammar-fused spec on a CPU session for greedy constrained
// requests. The loop used to leave its trailing token unforwarded at that exit, the n-gram loop's defect that on Metal
// changed every later turn (docs/measurements/spec-vs-batching-metal-2026-09-27.md §4). Turn 0 is capped at 6 tokens,
// short of the object, so it ends at max_tokens rather than at grammar completion.
func TestGrammarSpec_sessionTwoTurnsMatchPlain(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	tk, err := tokenizer.LoadGGUF(benchGGUFPath())
	if err != nil {
		t.Skipf("no tokenizer (%v)", err)
	}
	vocabBytes := constrain.TokenBytes(m.w.arch.VocabSize, tk.TokenText)
	encode := func(s string) []int { ids, _ := tk.Encode(s, false); return ids }
	schema := `{"type":"object","properties":{"location":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]}},"required":["location","unit"],"additionalProperties":false}`
	newMask := func() *constrain.Masker {
		g, gerr := constrain.JSONSchema([]byte(schema))
		if gerr != nil {
			t.Fatalf("JSONSchema: %v", gerr)
		}
		return constrain.NewMasker(g, vocabBytes, m.eosIDs).StopWhenComplete()
	}
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}
	p0 := encode("<|im_start|>user\nWeather for Paris in celsius as JSON matching the schema.<|im_end|>\n<|im_start|>assistant\n")

	turn := func(s *Session, spec bool, prompt []int, n int) []int {
		mask := newMask()
		if spec {
			ch, g, err := s.GenerateGrammarSpeculative(ctx, prompt, n, mask, &GrammarDrafter{Mask: mask, Encode: encode}, 8, greedy)
			if err != nil {
				t.Fatalf("GenerateGrammarSpeculative: %v", err)
			}
			out := collectTokens(ch)
			if g.Err() != nil {
				t.Fatalf("spec: %v", g.Err())
			}
			return out
		}
		sp := greedy
		sp.LogitProcessor = mask.Process
		ch, g := s.Generate(ctx, prompt, n, sp)
		out := collectTokens(ch)
		if g.Err() != nil {
			t.Fatalf("plain: %v", g.Err())
		}
		return out
	}
	run := func(spec bool) (out0, held, out1 []int) {
		s := m.NewSession(0)
		out0 = turn(s, spec, p0, 6)
		held = slices.Clone(s.Tokens())
		p1 := append(append(slices.Clone(p0), out0...), encode("<|im_end|>\n<|im_start|>user\nAnd for Rome?<|im_end|>\n<|im_start|>assistant\n")...)
		out1 = turn(s, spec, p1, 32)
		return
	}
	plain0, plainHeld, plain1 := run(false)
	spec0, specHeld, spec1 := run(true)
	if len(plain0) != 6 {
		t.Fatalf("plain turn 0 emitted %d tokens, want 6 (it must end at max_tokens for this test to mean anything)", len(plain0))
	}
	if !slices.Equal(plain0, spec0) {
		t.Fatalf("turn 0: spec %v, plain %v", spec0, plain0)
	}
	if !slices.Equal(plainHeld, specHeld) {
		t.Errorf("after turn 0 the session holds %d tokens under spec, %d under plain decode", len(specHeld), len(plainHeld))
	}
	if !slices.Equal(plain1, spec1) {
		t.Errorf("turn 1: spec %v, plain %v", spec1, plain1)
	}
}

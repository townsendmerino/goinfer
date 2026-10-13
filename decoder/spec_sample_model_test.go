package decoder

import (
	"context"
	"testing"
)

func TestNgramSampledFirstTokenMatchesPlain(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	ctx := context.Background()
	for _, seed := range []int64{1, 42, 1234} {
		sp := SamplingParams{Temperature: 0.8, Seed: seed} // no top-k/p/min-p → pure temperature
		for pi, prompt := range specPrompts {
			refCh, _ := m.Generate(ctx, prompt, 1, sp)
			ref := collectTokens(refCh)
			ch, g, err := m.GenerateNgramSpeculative(ctx, prompt, 1, &NgramDrafter{}, 8, sp)
			if err != nil {
				t.Fatalf("seed %d prompt %d: %v", seed, pi, err)
			}
			got := collectTokens(ch)
			if g.Err() != nil {
				t.Fatalf("seed %d prompt %d: stream err %v", seed, pi, g.Err())
			}
			if len(ref) != 1 || len(got) != 1 || ref[0] != got[0] {
				t.Fatalf("seed %d prompt %d: first token differs: plain %v vs sampled-spec %v", seed, pi, ref, got)
			}
		}
	}
}

func TestNgramSampledRejectsUnsupported(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	ctx := context.Background()
	// Penalties + bias are now supported — must NOT error.
	ok := []SamplingParams{
		{Temperature: 0.8, RepeatPenalty: 1.1},
		{Temperature: 0.8, PresencePenalty: 0.5, FrequencyPenalty: 0.5},
		{Temperature: 0.8, LogitBias: map[int]float32{1: 5}},
	}
	for i, sp := range ok {
		ch, g, err := m.GenerateNgramSpeculative(ctx, specPrompts[0], 4, &NgramDrafter{}, 8, sp)
		if err != nil {
			t.Errorf("supported case %d: unexpected error %v", i, err)
			continue
		}
		collectTokens(ch)
		if g.Err() != nil {
			t.Errorf("supported case %d: stream err %v", i, g.Err())
		}
	}
	// A LogitProcessor (constrained/tool decoding) is still rejected.
	bad := SamplingParams{Temperature: 0.8, LogitProcessor: func([]int, []float32) {}}
	if _, _, err := m.GenerateNgramSpeculative(ctx, specPrompts[0], 4, &NgramDrafter{}, 8, bad); err == nil {
		t.Error("LogitProcessor: expected rejection, got nil error")
	}
}

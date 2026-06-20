package decoder

import (
	"context"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestNgramSampledHarness re-measures acceptance under sampling to test the
// α̅≫acc hypothesis from the greedy run: greedy accepts a draft token only when it
// is the target's argmax, so realized acceptance sits far below α̅ (the mean
// per-position 1−TV) wherever the draft is "strong but not top". Sampled rejection
// accepts with probability p(x), so realized acceptance should rise to ≈ α̅ —
// capturing that mass — which means more committed tokens per verify on the
// workloads where greedy left it on the table (e.g. codeedit).
//
//	go test ./decoder -run TestNgramSampledHarness -v
func TestNgramSampledHarness(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	tk, err := tokenizer.LoadGGUF(benchGGUFPath())
	if err != nil {
		t.Skipf("no tokenizer (%v)", err)
	}
	const maxTok = 128
	const K = 8
	ctx := context.Background()
	modes := []struct {
		name string
		sp   SamplingParams
	}{
		{"greedy", SamplingParams{Temperature: 0}},
		{"sampled", SamplingParams{Temperature: 0.7, TopP: 0.95, Seed: 1234}},
	}

	// eval-acc (accepted/evaluated) is comparable to α̅; acc (accepted/proposed)
	// also charges the untested post-reject tail, so acc≪eval-acc exposes chain
	// breaking. tok/v is the throughput that actually matters.
	t.Logf("%-11s %-8s %6s  %8s  %6s  %6s", "workload", "mode", "α̅", "eval-acc", "acc", "tok/v")
	for _, w := range specWorkloads {
		prompt, err := tk.Encode(w.prompt, true)
		if err != nil {
			t.Fatalf("%s: encode: %v", w.name, err)
		}
		for _, md := range modes {
			col := NewTraceCollector(nil)
			ch, g, err := m.genNgram(ctx, prompt, maxTok, &NgramDrafter{}, K, md.sp, col.Record, nil)
			if err != nil {
				t.Fatalf("%s %s: %v", w.name, md.name, err)
			}
			collectTokens(ch)
			if g.Err() != nil {
				t.Fatalf("%s %s: stream err %v", w.name, md.name, g.Err())
			}
			t.Logf("%-11s %-8s %6.3f  %8.3f  %6.3f  %6.2f",
				w.name, md.name, col.MeanAccept(), g.Spec.EvalAcceptanceRate(), g.Spec.AcceptanceRate(), g.Spec.TokensPerRound())
		}
	}
}

package decoder

import (
	"context"
	"testing"
)

// parityPrompt + parityWant pin the 0.5B's greedy continuation so any perf
// refactor (fusion, scratch reuse, kernel changes) that moves numerics is caught
// immediately. Greedy + fixed prompt → fully deterministic. Regenerate the want
// list ONLY when a numerics change is intentional and parity-reviewed.
var parityPrompt = []int{785, 264, 6573, 311, 1438, 279, 2038, 25}

var parityWant = []int{4710, 73594, 12669, 198, 750, 1438, 4136, 3932, 262, 671, 1096, 729, 686, 1438, 279, 2038, 198, 262, 1494, 271, 2, 7143, 279, 729}

// TestDecodeParity greedily continues parityPrompt and checks the token ids
// against parityWant. Skips without the model asset.
func TestDecodeParity(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GINFER_PREQUANT_GGUF", err)
	}
	const n = 24
	out, gen := m.Generate(context.Background(), parityPrompt, n, SamplingParams{Temperature: 0})
	var got []int
	for tok := range out {
		got = append(got, tok)
	}
	if err := gen.Err(); err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(parityWant) == 0 {
		t.Logf("CAPTURE parityWant = %#v", got)
		t.Skip("parityWant empty — capture run")
	}
	if len(got) != len(parityWant) {
		t.Fatalf("got %d tokens, want %d: %v", len(got), len(parityWant), got)
	}
	for i := range got {
		if got[i] != parityWant[i] {
			t.Fatalf("token %d: got %d want %d\nfull: %v", i, got[i], parityWant[i], got)
		}
	}
}

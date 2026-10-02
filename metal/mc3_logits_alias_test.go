//go:build darwin

package metal

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMC3_soloLogitsSurviveAnotherGeneration is E-C01 (docs/audit-metal-2026-09-30.md). metalResident.Forward returns
// the resident's one host logits buffer, reused by every call. Under MC3 a token served alone (the solo path) hands that
// buffer to its generation, which reads it AFTER leaving the resident: in its LogitProcessor and its sampler. Another
// generation's solo token rewrites the buffer as soon as it runs. Generation A here takes 30 ms in its processor while
// generation B, 8 ms in its own, decodes alongside; both are on the full-logits path (a processor rules out the
// on-device argmax). A's greedy tokens must equal A's alone. Before the fix they did not: A sampled from B's logits,
// from its first token (the prompt's seed, handed out by mc3Prefill) and in decode (a solo token's logits).
func TestMC3_soloLogitsSurviveAnotherGeneration(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	m, err := decoder.Load(writeMC3Fixture(t, 4096), decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024, ResidentKVSlots: 2})
	if err != nil {
		t.Fatalf("load the MC3 fixture on Metal: %v", err)
	}
	defer m.Close()
	if n := m.EnableResidentConcurrency(2); n != 2 {
		t.Fatalf("EnableResidentConcurrency(2) = %d: MC3 is not on, so there is no solo path to test", n)
	}
	ctx := context.Background()
	slow := decoder.SamplingParams{LogitProcessor: func(_ []int, _ []float32) { time.Sleep(30 * time.Millisecond) }}
	// B's processor takes 8 ms. Its tokens then run solo while A is in its own processor, and A's run solo while B is
	// in its: the interleaving both halves of the fix guard. With an instant processor B always has a token pending
	// when A submits, A's tokens join batched steps (whose rows are copied), and only the prompt's seed is exposed.
	busy := decoder.SamplingParams{LogitProcessor: func(_ []int, _ []float32) { time.Sleep(8 * time.Millisecond) }}
	promptA, promptB := make([]int, 16), make([]int, 16)
	for i := range promptA {
		promptA[i], promptB[i] = 100+i*37, 9000+i*53
	}
	gen := func(prompt []int, n int, sp decoder.SamplingParams) []int {
		ch, g := m.Generate(ctx, prompt, n, sp)
		var out []int
		for id := range ch {
			out = append(out, id)
		}
		if err := g.Err(); err != nil {
			t.Errorf("generate: %v", err)
		}
		return out
	}
	alone := gen(promptA, 12, slow)
	var withB []int
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); gen(promptB, 60, busy) }()
	go func() { defer wg.Done(); time.Sleep(20 * time.Millisecond); withB = gen(promptA, 12, slow) }()
	wg.Wait()
	t.Logf("A alone %v; A beside B %v", alone, withB)
	if len(alone) != 12 || !slices.Equal(alone, withB) {
		t.Fatalf("generation A's greedy tokens changed when B decoded alongside it: alone %v, beside B %v", alone, withB)
	}
}

package decoder

import (
	"context"
	"os"
	"runtime"
	"testing"
)

// longPromptIDs is a deterministic ≥512-token prompt. The CONTENT does not matter; the LENGTH
// does — it must sit above fastAttnMinPrompt so this golden exercises the f32 prefill path that
// every other forward golden misses.
func longPromptIDs(n int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = 700 + (i*7919)%9000 // fixed sequence, no RNG, no model-specific vocabulary
	}
	return p
}

// longPromptFastWant is the greedy continuation of longPromptIDs(768) on the bench model through
// the DEFAULT (f32) prefill path. Regenerate ONLY when a numerics change to that path is
// intentional and reviewed — the same contract parityWant carries for the exact path.
//
// KEYED BY GOARCH, and that is a FINDING, not boilerplate. Measured 2026-09-01 on the same
// checkpoint and commit: arm64 and amd64 diverge at the FIRST generated token
// (11 714 279 ... vs 13 715 522 ...). The exact kernel does not do this — parityWant is one
// list for both — so cross-arch token reproducibility is something the f32 default gives up,
// beyond the "decode != prefill" the flag documents. gc fuses x*y+z into FMA on arm64 and not
// on amd64 (docs/parity-coverage-policy.md, "CPU reference is arch-scoped"), and f32 attention
// has no f64 accumulator to absorb the difference.
//
// A single shared list here would have been a permanently red gate on one of the two CI runners.
var longPromptFastWant = map[string][]int{
	"arm64": {11, 714, 279, 3491, 374, 429, 279, 2038, 374, 537, 3238, 438, 3601, 13, 576, 1465},
	"amd64": {13, 715, 522, 2599, 397, 522, 1551, 397, 522, 2599, 397, 522, 1551, 397, 522, 2599},
}

// TestLongPromptFast_forwardParity closes the coverage hole that flipping --cpu-fast-attention's
// default exposed: every other forward golden uses a prompt SHORTER than fastAttnMinPrompt, so
// they all take the exact kernel and the f32 path — the shipped default — had no golden at all.
//
// WHY THAT MATTERED CONCRETELY, not hypothetically. scripts/refresh_parity_hashes.sh is the
// sanctioned release valve for a hashed-core edit, and it reported the default flip as a
// "non-numeric core refresh" — true of everything the goldens touched, false of the change. A
// green that covers only the unchanged side of an edit is the exact failure Q1 documents.
//
// THE NAME ENDS IN _forwardParity DELIBERATELY: that is what the refresh script's selector
// matches ((_forwardParity|_logitParity|_textParity)$|^TestGGUF_.*_parity$). A gate that the
// release valve does not run is a gate that does not protect the release valve.
func TestLongPromptFast_forwardParity(t *testing.T) {
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	const K = 768 // > fastAttnMinPrompt (512), so this is the f32 path
	if K < fastAttnMinPrompt {
		t.Fatalf("K=%d is below the floor %d — this golden would silently test the exact path", K, fastAttnMinPrompt)
	}
	if !m.canBatchN(K) {
		t.Skip("model has no batched prefill")
	}
	prompt := longPromptIDs(K)

	// Default path (no env override): this is what a user gets.
	os.Unsetenv("GOINFER_CPU_FAST_ATTENTION")
	out, gen := m.Generate(context.Background(), prompt, 16, SamplingParams{Temperature: 0})
	var got []int
	for id := range out {
		got = append(got, id)
	}
	if gen.Err() != nil {
		t.Fatalf("generate: %v", gen.Err())
	}
	want, ok := longPromptFastWant[runtime.GOARCH]
	if !ok {
		t.Skipf("no f32 golden recorded for GOARCH=%s — record one rather than comparing against another arch's", runtime.GOARCH)
	}
	for i := range want {
		if i >= len(got) || got[i] != want[i] {
			t.Fatalf("f32 prefill continuation drifted at %d (GOARCH=%s):\n got %v\nwant %v", i, runtime.GOARCH, got, want)
		}
	}
	// NON-VACUITY: prove this golden is on the f32 side. If the floor moved, or the default were
	// reverted, the assertions above would still pass while testing the exact kernel — a golden
	// that silently changes which path it covers is worse than no golden.
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0")
	exOut, _ := m.Generate(context.Background(), prompt, 16, SamplingParams{Temperature: 0})
	var exact []int
	for id := range exOut {
		exact = append(exact, id)
	}
	same := len(exact) == len(got)
	for i := range exact {
		if i < len(got) && exact[i] != got[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("the default and the exact kernel produced identical output at K=768 — this golden " +
			"is NOT covering the f32 path (floor raised above 768, or the default reverted?)")
	}
}

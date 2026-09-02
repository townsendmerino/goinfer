package decoder

import (
	"context"
	"os"
	"strings"
	"testing"
)

// TestSessionFastAttnDivergence — THE TEST forwardn.go:44 HAS BEEN CITING ALL ALONG.
//
// It did not exist anywhere in the tree. The doc comment above cpuFastAttention says
// "TestSessionFastAttnDivergence pins the new behaviour", which is the exact "a doc comment
// claiming coverage is not coverage" class CLAUDE.md describes — one file away from where that
// rule was written. Whoever audited the claim would find a plausible name, match it to the
// sentence, and stop.
//
// What it now pins is the honest version of the contract: with the fast kernel on (the
// default), a prompt at or above fastAttnMinPrompt gives DIFFERENT logits from the exact
// kernel — the split-invariance loss the comment describes and docs/server.md used to deny —
// and below the floor the two are identical, because the floor turns the fast path off.
func TestSessionFastAttnDivergence(t *testing.T) {
	const fixture = "../testdata/llama-tiny"
	if _, err := os.Stat(fixture); err != nil {
		t.Skipf("no fixture at %s: %v", fixture, err)
	}
	m, err := Load(fixture, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	_, nL, _, nKV, hd, _, vocab := m.Dims()

	prefill := func(prompt []int, fast bool) []float32 {
		t.Helper()
		t.Setenv("GOINFER_CPU_FAST_ATTENTION", map[bool]string{true: "1", false: "0"}[fast])
		cache := NewKVCache(nL, nKV, hd, 0, len(prompt)+8)
		lg, err := m.prefillLogits(context.Background(), prompt, cache)
		if err != nil {
			t.Fatalf("prefillLogits(fast=%v, %d tokens): %v", fast, len(prompt), err)
		}
		return append([]float32(nil), lg...)
	}
	seq := func(n int) []int {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = (i*131 + 7) % vocab
		}
		return ids
	}
	differs := func(a, b []float32) bool {
		if len(a) != len(b) {
			return true
		}
		for i := range a {
			if a[i] != b[i] {
				return true
			}
		}
		return false
	}

	// BELOW THE FLOOR: identical, because fastAttnMinPrompt turns the fast path off. This is
	// the half that makes the test above meaningful — without it, a build where the flag did
	// nothing at all would pass the divergence check by never diverging.
	t.Run("below the floor the kernels agree", func(t *testing.T) {
		short := seq(fastAttnMinPrompt - 1)
		if differs(prefill(short, true), prefill(short, false)) {
			t.Errorf("a %d-token prompt differs between kernels — the fastAttnMinPrompt floor "+
				"is meant to keep short prompts on the exact path", len(short))
		}
	})

	// AT AND ABOVE THE FLOOR: the fast kernel engages, and the result is NOT bit-identical.
	// That is the accepted 2026-08-31 behaviour; what was missing is anything pinning it, so a
	// change that silently made the flag inert — or silently applied it everywhere — would go
	// unnoticed in both directions.
	t.Run("at the floor the kernels diverge", func(t *testing.T) {
		long := seq(fastAttnMinPrompt)
		if !differs(prefill(long, true), prefill(long, false)) {
			t.Errorf("a %d-token prompt is bit-identical between kernels. Either the fast path "+
				"no longer engages at the floor (the flag is inert, and its measured win is "+
				"gone), or it became exact — in which case docs/server.md's reuse contract and "+
				"forwardn.go's NOT-SPLIT-INVARIANT note should say so", len(long))
		}
	})
}

// M-07, the EAGLE half. EAGLE's contract is that it is token-identical to plain greedy, and
// that rests on both paths building the SAME KV for the same prompt. They did not: EAGLE
// prefilled through forwardN, which is hard-wired to the exact kernel because speculative
// VERIFY needs it, while Generate's prefillLogits uses cpuFastAttention(). Above the 512-token
// floor the two KVs differ, so the emitted streams can diverge at temperature 0.
// TestEagleSpecParity uses a ~25-token prompt — comfortably under the floor, so it could not
// see this.
//
// Asserted on the source: the divergence needs a real EAGLE head and a >512-token prompt to
// show up end-to-end, and what actually has to hold is the narrow structural claim that
// prefill and verify make DIFFERENT choices, with prefill matching prefillLogits.
func TestEagle_prefillsWithTheSameKernelAsGenerate(t *testing.T) {
	src, err := os.ReadFile("spec_eagle.go")
	if err != nil {
		t.Fatalf("read spec_eagle.go: %v", err)
	}
	var prefill, exact int
	for _, ln := range strings.Split(string(src), "\n") {
		s := strings.TrimSpace(ln)
		if strings.HasPrefix(s, "//") || !strings.Contains(s, "captureN(") {
			continue
		}
		switch {
		case strings.Contains(s, "captureN(prompt, cpuFastAttention())"):
			prefill++
		case strings.Contains(s, ", false)"):
			exact++
		default:
			t.Errorf("a captureN call makes neither choice explicitly: %s", s)
		}
	}
	// Two prefill sites (the two EAGLE loops), four verify/commit/correction sites.
	if prefill != 2 {
		t.Errorf("%d EAGLE prefill(s) use cpuFastAttention(), want 2 — a prefill on the exact "+
			"kernel builds different KV from Generate's, and EAGLE's token-identical-to-greedy "+
			"claim depends on them matching (M-07)", prefill)
	}
	if exact != 4 {
		t.Errorf("%d captureN call(s) pin the exact kernel, want 4 — speculative verify must run "+
			"acc64 on both arms or its equality argument collapses; only PREFILL should follow "+
			"the operator's flag", exact)
	}
}

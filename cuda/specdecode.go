//go:build cuda

package cuda

// Speculative decoding (D1) — n-gram / prompt-lookup drafting + batched verify. Greedy-lossless by
// construction: the verify's per-position logits (PrefillLastN) are bit-identical to a sequential
// Forward at each position, so the accepted tokens are exactly the sequential greedy tokens. The
// drafter is FREE (no draft model — a context lookup), so the whole cost is the batched M=k verify,
// which amortizes the weight read across k tokens (measured 2.5–3.6× cheaper than k decodes at k=4–8,
// TestSpecVerifyCeiling). See docs/ollama-chase.md §D1.

// ngramDraft proposes up to k tokens by finding the most recent earlier occurrence of the last
// ctxLen tokens in hist and returning the tokens that followed it ("prompt lookup"). Returns nil when
// there's no match (the caller then plain-decodes one token). Deterministic; drafting quality only
// affects speed, never correctness (a wrong draft is simply rejected by verify).
func ngramDraft(hist []int, k, ctxLen int) []int {
	n := len(hist)
	if ctxLen < 1 || n < ctxLen+1 {
		return nil
	}
	suffix := hist[n-ctxLen:]
	// Most recent earlier match wins (locality: recent context predicts best).
	for i := n - ctxLen - 1; i >= 0; i-- {
		ok := true
		for j := 0; j < ctxLen; j++ {
			if hist[i+j] != suffix[j] {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}
		src := i + ctxLen // tokens that followed the match
		m := k
		if src+m > n {
			m = n - src
		}
		if m <= 0 {
			return nil
		}
		return append([]int(nil), hist[src:src+m]...)
	}
	return nil
}

// SpecStats reports one speculative-decode run.
type SpecStats struct {
	Generated   int // tokens produced (excluding the seed)
	Rounds      int // verify rounds
	VerifyToks  int // total tokens fed to the batched verify (the work done)
	Drafted     int // tokens proposed by the drafter
	Accepted    int // drafted tokens accepted (excludes the per-round bonus)
	PlainRounds int // rounds with no draft (fell back to a 1-token step)
}

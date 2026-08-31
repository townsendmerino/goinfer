package decoder

import (
	"encoding/json"
	"os"
	"slices"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// 02 "next step" — Step 2: does CANDIDATE SCORING (frequency vs most-recent)
// beat the shipped policy, isolated from the DATA STRUCTURE?
//
// This needs no model. In greedy mode a model-free drafter's acceptance is
// exactly "does the proposed token equal the token the target actually emitted",
// and the emitted tokens are recorded by step 1 (losslessness guarantees the
// speculative and plain streams are the same tokens). So every policy can be
// replayed exactly against the recorded stream, on the SAME rounds, which is the
// paired comparison the pre-registration asks for.
//
// Structure is held FIXED across arms — every policy selects the same longest
// matching suffix L — so any difference is scoring alone.
// ---------------------------------------------------------------------------

// candPolicy proposes up to k tokens given the context, or nil on a miss. It
// also reports the matched suffix length for the match-length histogram.
type candPolicy func(ctx []int, k int) (prop []int, matchLen int)

// occurrencesOfLongestSuffix finds the longest suffix (<= maxM, >= minM) of ctx
// that occurs earlier, and returns every earlier start offset for it. This is the
// shared "structure" half; the arms below differ only in what they do with it.
func occurrencesOfLongestSuffix(ctx []int, minM, maxM int) (occs []int, L int) {
	n := len(ctx)
	hi := min(maxM, n-1)
	for L = hi; L >= minM; L-- {
		pat := ctx[n-L:]
		for s := n - L - 1; s >= 0; s-- {
			if slices.Equal(ctx[s:s+L], pat) {
				occs = append(occs, s) // collected most-recent-first
			}
		}
		if len(occs) > 0 {
			return occs, L
		}
	}
	return nil, 0
}

// policyMostRecent is the SHIPPED policy: longest suffix, most recent earlier hit.
func policyMostRecent(ctx []int, k int) ([]int, int) {
	occs, L := occurrencesOfLongestSuffix(ctx, 2, 16)
	if len(occs) == 0 {
		return nil, 0
	}
	s := occs[0] // most recent
	cont := ctx[s+L:]
	return slices.Clone(cont[:min(k, len(cont))]), L
}

// policyFrequency is SuffixDecoding's signal: among ALL earlier occurrences of the
// same matched suffix, walk the most frequent continuation token at each depth
// (ties → the most recent occurrence's choice). This is a frequency-scored PATH,
// not just a frequency-scored first token.
func policyFrequency(ctx []int, k int) ([]int, int) {
	occs, L := occurrencesOfLongestSuffix(ctx, 2, 16)
	if len(occs) == 0 {
		return nil, 0
	}
	active := make([]int, len(occs))
	for i, s := range occs {
		active[i] = s + L // read cursor into ctx
	}
	var path []int
	for range k {
		counts := map[int]int{}
		firstSeen := map[int]int{} // preserve most-recent-first tie-break
		for i, p := range active {
			if p >= len(ctx) {
				continue
			}
			t := ctx[p]
			counts[t]++
			if _, ok := firstSeen[t]; !ok {
				firstSeen[t] = i
			}
		}
		if len(counts) == 0 {
			break
		}
		best, bestN, bestSeen := -1, -1, 1<<30
		for t, c := range counts {
			if c > bestN || (c == bestN && firstSeen[t] < bestSeen) {
				best, bestN, bestSeen = t, c, firstSeen[t]
			}
		}
		path = append(path, best)
		var next []int
		for _, p := range active {
			if p < len(ctx) && ctx[p] == best {
				next = append(next, p+1)
			}
		}
		active = next
	}
	return path, L
}

// policyMostRecentDeep is the shipped policy with the MaxMatch probe cap raised
// 16 -> 64. Added because the step-1 histogram showed 14 of 20 hits sitting at
// EXACTLY 16 on code-continue-2 — matches pinned at the cap are matches being
// truncated, and match length is the alpha-hat signal (spec_ngram.go's
// ngramAlphaAnchors tops out at 16 => 0.97). This asks whether the cap is
// costing accepted tokens or only mis-reporting confidence.
func policyMostRecentDeep(ctx []int, k int) ([]int, int) {
	occs, L := occurrencesOfLongestSuffix(ctx, 2, 64)
	if len(occs) == 0 {
		return nil, 0
	}
	s := occs[0]
	cont := ctx[s+L:]
	return slices.Clone(cont[:min(k, len(cont))]), L
}

// policyOracle is the CEILING for candidate selection at fixed structure: among all
// occurrences of the matched suffix, pick the one whose continuation would actually
// have been accepted longest. It cheats (it reads the future) and exists only to
// bound item 2 — if the oracle barely beats most-recent, NO scoring scheme can.
func policyOracle(ctx []int, k int, truth []int) ([]int, int) {
	occs, L := occurrencesOfLongestSuffix(ctx, 2, 16)
	if len(occs) == 0 {
		return nil, 0
	}
	best, bestAcc := []int(nil), -1
	for _, s := range occs {
		cont := ctx[s+L:]
		c := cont[:min(k, len(cont))]
		acc := 0
		for acc < len(c) && acc < len(truth) && c[acc] == truth[acc] {
			acc++
		}
		if acc > bestAcc {
			bestAcc, best = acc, slices.Clone(c)
		}
	}
	return best, L
}

// tailCycle is the SECOND, INDEPENDENT loop detector, added when the first one
// turned out to be ambiguous on code. distinct-trigram measures REPETITIVENESS,
// and real Go source is legitimately repetitive (`float64`, `\n\t`, receiver
// names) — the shipped step-1 code trace scored 0.715 against a 0.70 bar written
// for PROSE coherence. Conflating "repetitive" with "looping" would either
// discard the exact traffic this task is about, or pass a genuinely degenerate
// trace. So this checks the actual failure mode instead: is the TAIL periodic?
//
// It returns the smallest period p (<= 32) such that the last `repeats*p` tokens
// are exactly periodic with period p, and how many repeats that runs for. A real
// infinite loop shows a small p repeating many times; repetitive-but-progressing
// code does not.
//
// Deliberately NOT a replacement for the pre-registered trigram rule: that rule
// stays binding, this one is reported beside it, and a DISAGREEMENT between them
// is itself the finding.
func tailCycle(toks []int) (period, repeats int) {
	n := len(toks)
	for p := 1; p <= 32 && p*2 <= n; p++ {
		r := 0
		for (r+1)*p <= n {
			ok := true
			for j := range p {
				if toks[n-(r+1)*p+j] != toks[n-p+j] {
					ok = false
					break
				}
			}
			if !ok {
				break
			}
			r++
		}
		if r >= 4 { // 4+ consecutive identical blocks at the tail
			return p, r
		}
	}
	return 0, 0
}

type replayResult struct {
	Rounds, Hits, Proposed, Accepted int
	MatchLenHist                     map[int]int
}

func (r replayResult) hitRate() float64 {
	if r.Rounds == 0 {
		return 0
	}
	return float64(r.Hits) / float64(r.Rounds)
}
func (r replayResult) acceptedPerRound() float64 {
	if r.Rounds == 0 {
		return 0
	}
	return float64(r.Accepted) / float64(r.Rounds)
}
func (r replayResult) alphaOnHits() float64 {
	if r.Proposed == 0 {
		return 0
	}
	return float64(r.Accepted) / float64(r.Proposed)
}

// replay walks the recorded stream exactly as genNgramInto does: each round drafts
// from the context ending at cur, accepts the leading run that matches what the
// target actually emitted, and advances by 1 (cur) + accepted.
func replay(toks []int, promptLen, k int, pol candPolicy, oracle bool) replayResult {
	res := replayResult{MatchLenHist: map[int]int{}}
	i := promptLen // index of `cur`
	for i < len(toks)-1 {
		lookup := toks[:i+1]
		truth := toks[i+1:]
		var prop []int
		var L int
		if oracle {
			prop, L = policyOracle(lookup, k, truth)
		} else {
			prop, L = pol(lookup, k)
		}
		res.Rounds++
		if len(prop) > 0 {
			res.Hits++
			res.Proposed += len(prop)
			res.MatchLenHist[L]++
		}
		acc := 0
		for acc < len(prop) && acc < len(truth) && prop[acc] == truth[acc] {
			acc++
		}
		res.Accepted += acc
		i += 1 + acc
	}
	return res
}

// TestSuffixProbe_step2 is the pre-registered offline replay. Input is step 1's
// artifact; no model is loaded.
//
// Run: GOINFER_SPEC_SUFFIX_IN=<step1.json> go test ./decoder -run TestSuffixProbe_step2 -v
func TestSuffixProbe_step2(t *testing.T) {
	path := os.Getenv("GOINFER_SPEC_SUFFIX_IN")
	if path == "" {
		t.Skip("set GOINFER_SPEC_SUFFIX_IN to step 1's artifact")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read step-1 artifact: %v", err)
	}
	var recs []struct {
		Name        string  `json:"name"`
		PromptToks  int     `json:"prompt_toks"`
		Excluded    bool    `json:"excluded_as_looping"`
		Accepted    int     `json:"accepted"`
		Rounds      int     `json:"rounds"`
		DistinctTri float64 `json:"distinct_trigram"`
		Tokens      []int   `json:"tokens"`
	}
	if err := json.Unmarshal(b, &recs); err != nil {
		t.Fatalf("parse: %v", err)
	}

	const K = 8
	t.Logf("%-18s %-12s %6s %7s %8s %9s %10s", "workload", "policy", "rounds", "hitrate", "alpha", "acc/round", "vs shipped")
	for _, r := range recs {
		gen := r.Tokens[min(r.PromptToks, len(r.Tokens)):]
		cp, cr := tailCycle(gen)
		cyc := "no tail cycle"
		if cp > 0 {
			cyc = "TAIL CYCLE period=" + strconvFormat(float64(cp)) + " repeats=" + strconvFormat(float64(cr))
		}
		// The two detectors are reported together on purpose; disagreement is a finding.
		t.Logf("  [loop guards %s] distinct-trigram=%.3f (bar 0.70, %s) | %s",
			r.Name, r.DistinctTri, map[bool]string{true: "EXCLUDES", false: "passes"}[r.DistinctTri < 0.70], cyc)
		if r.Excluded {
			t.Logf("%-18s EXCLUDED by the pre-registered trigram rule", r.Name)
			continue
		}
		if len(r.Tokens) == 0 {
			continue
		}
		base := replay(r.Tokens, r.PromptToks, K, policyMostRecent, false)
		freq := replay(r.Tokens, r.PromptToks, K, policyFrequency, false)
		orac := replay(r.Tokens, r.PromptToks, K, nil, true)

		// Cross-check: the replay must reproduce the LIVE run's accepted count.
		// If it does not, the replay model is wrong and nothing below is usable.
		if r.Rounds > 0 {
			t.Logf("  [cross-check %s] live accepted=%d rounds=%d | replay accepted=%d rounds=%d",
				r.Name, r.Accepted, r.Rounds, base.Accepted, base.Rounds)
		}
		deep := replay(r.Tokens, r.PromptToks, K, policyMostRecentDeep, false)
		for _, a := range []struct {
			name string
			res  replayResult
		}{{"most-recent", base}, {"frequency", freq}, {"deep-cap64", deep}, {"oracle", orac}} {
			delta := ""
			if a.name != "most-recent" && base.acceptedPerRound() > 0 {
				delta = formatPct(a.res.acceptedPerRound()/base.acceptedPerRound() - 1)
			}
			t.Logf("%-18s %-12s %6d %7.3f %8.3f %9.3f %10s",
				r.Name, a.name, a.res.Rounds, a.res.hitRate(), a.res.alphaOnHits(), a.res.acceptedPerRound(), delta)
		}
		t.Logf("%-18s match-len histogram (most-recent): %v", r.Name, base.MatchLenHist)
	}
}

func formatPct(f float64) string {
	s := "+"
	if f < 0 {
		s = ""
	}
	return s + strconvFormat(f*100) + "%"
}

func strconvFormat(f float64) string {
	b, _ := json.Marshal(roundTo(f, 1))
	return string(b)
}

func roundTo(f float64, places int) float64 {
	p := 1.0
	for range places {
		p *= 10
	}
	return float64(int(f*p+sign(f)*0.5)) / p
}

func sign(f float64) float64 {
	if f < 0 {
		return -1
	}
	return 1
}

// TestSuffixProbe_draftCost measures what the SHIPPED linear scan actually costs,
// as a share of a round. Item 1's premise is that a suffix automaton's sublinear
// lookup beats an O(maxM * n) backward scan "that does not degrade as context
// grows" — but a lookup that is already a rounding error against a target forward
// pass cannot be worth a new data structure, however much better its asymptotics
// are. This measures the constant instead of assuming the asymptote matters.
//
// Run: GOINFER_SPEC_SUFFIX_IN=<step1.json> go test ./decoder -run TestSuffixProbe_draftCost -v
func TestSuffixProbe_draftCost(t *testing.T) {
	path := os.Getenv("GOINFER_SPEC_SUFFIX_IN")
	if path == "" {
		t.Skip("set GOINFER_SPEC_SUFFIX_IN to step 1's artifact")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var recs []struct {
		Name       string  `json:"name"`
		PromptToks int     `json:"prompt_toks"`
		Excluded   bool    `json:"excluded_as_looping"`
		Rounds     int     `json:"rounds"`
		OffMs      float64 `json:"off_ms"`
		GenToks    int     `json:"gen_toks"`
		Tokens     []int   `json:"tokens"`
	}
	if err := json.Unmarshal(b, &recs); err != nil {
		t.Fatalf("parse: %v", err)
	}
	for _, r := range recs {
		if r.Excluded || len(r.Tokens) == 0 {
			continue
		}
		d := &NgramDrafter{}
		// Time Draft over the real contexts this run actually presented.
		const reps = 50
		t0 := time.Now()
		n := 0
		for range reps {
			for i := r.PromptToks; i < len(r.Tokens)-1; i += 8 {
				d.Draft(r.Tokens[:i+1], 8)
				n++
			}
		}
		perDraft := float64(time.Since(t0).Nanoseconds()) / float64(n) / 1000 // µs
		// One plain decode step, from the `off` arm: off_ms / generated tokens.
		perStepUs := r.OffMs * 1000 / float64(max(r.GenToks, 1))
		t.Logf("%-18s ctx≈%d  Draft = %8.1f µs   target step = %9.1f µs   drafter share = %.4f%%",
			r.Name, len(r.Tokens), perDraft, perStepUs, 100*perDraft/perStepUs)
	}
}

// TestSuffixProbe_matchProvenance answers item 3 (SCOPE OF HISTORY) directly:
// when the drafter finds a match, WHERE is the matched occurrence?
//
//   - in the PROMPT  — which, on goinfer's stateless /v1/chat/completions surface,
//     IS the re-sent cross-request history: the client sends the whole transcript
//     every turn, so prior turns are already indexed by the shipped drafter.
//   - in the generation's OWN output — self-repetition within this response.
//   - no match at all.
//
// This is the quantity item 3's premise turns on. A cross-request suffix tree can
// only add supply on the MISS rounds, and only for patterns that appeared in an
// earlier request WITHOUT being re-sent — which on a stateless surface means a
// DIFFERENT conversation, not an earlier turn of this one.
//
// Note the bias direction, because it matters for the excluded traces: a looping
// generation manufactures long recent SELF-matches, and longest-suffix-wins means
// those outrank prompt matches. So looping DEFLATES the prompt share — the prompt
// percentages below are lower bounds on a looping trace, never inflated by it.
//
// Run: GOINFER_SPEC_SUFFIX_IN=<step1.json> go test ./decoder -run TestSuffixProbe_matchProvenance -v
func TestSuffixProbe_matchProvenance(t *testing.T) {
	path := os.Getenv("GOINFER_SPEC_SUFFIX_IN")
	if path == "" {
		t.Skip("set GOINFER_SPEC_SUFFIX_IN to step 1's artifact")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var recs []struct {
		Name       string `json:"name"`
		PromptToks int    `json:"prompt_toks"`
		Excluded   bool   `json:"excluded_as_looping"`
		Tokens     []int  `json:"tokens"`
	}
	if err := json.Unmarshal(b, &recs); err != nil {
		t.Fatalf("parse: %v", err)
	}
	for _, r := range recs {
		if len(r.Tokens) == 0 {
			continue
		}
		var inPrompt, inGen, miss int
		for i := r.PromptToks; i < len(r.Tokens)-1; i++ {
			occs, L := occurrencesOfLongestSuffix(r.Tokens[:i+1], 2, 16)
			switch {
			case len(occs) == 0:
				miss++
			case occs[0]+L <= r.PromptToks:
				inPrompt++ // match lies wholly in the re-sent history
			default:
				inGen++
			}
		}
		tot := max(inPrompt+inGen+miss, 1)
		note := ""
		if r.Excluded {
			note = "  [trace excluded as looping — prompt share is a LOWER BOUND]"
		}
		t.Logf("%-18s prompt=%4d (%5.1f%%)  own-gen=%4d (%5.1f%%)  miss=%4d (%5.1f%%)%s",
			r.Name, inPrompt, 100*float64(inPrompt)/float64(tot),
			inGen, 100*float64(inGen)/float64(tot), miss, 100*float64(miss)/float64(tot), note)
	}
}

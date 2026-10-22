package tokenizer

import (
	"math/rand"
	"slices"
	"strings"
	"testing"
)

// naiveMergeSymbols is the pre-M28 O(n²) reference: rescan for the globally
// lowest-rank pair (leftmost on a tie) and merge, repeat. mergeSymbols replaced it
// with an O(n log n) heap; this pins them byte-for-byte.
func naiveMergeSymbols(pr map[bigram]int32, syms []string) []string {
	syms = append([]string(nil), syms...)
	for len(syms) >= 2 {
		const maxRank = int32(1<<31 - 1)
		bestRank, bestI := maxRank, -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := pr[bigram{syms[i], syms[i+1]}]; ok && r < bestRank {
				bestRank, bestI = r, i
			}
		}
		if bestI < 0 {
			break
		}
		syms[bestI] += syms[bestI+1]
		syms = append(syms[:bestI+1], syms[bestI+2:]...)
	}
	return syms
}

// TestMergeSymbols_matchesNaive gates M28's heap rewrite: over many random inputs
// with a tie-heavy synthetic merge table (so ties, overlapping pairs, and cascades
// all occur), the heap merge must equal the naive rescan exactly — including the
// leftmost-on-a-tie choice and stale-candidate handling.
func TestMergeSymbols_matchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := range 3000 {
		n := 2 + rng.Intn(7) // up to 8 single-char symbols
		syms := make([]string, n)
		for i := range syms {
			syms[i] = string(rune('a' + rng.Intn(3))) // alphabet {a,b,c} ⇒ frequent repeats
		}
		// Rank every ordered pair of substrings that could ever become adjacent. Small
		// rank range ⇒ many ties; ~half the pairs unmergeable.
		joined := strings.Join(syms, "")
		var subs []string
		for i := 0; i < len(joined); i++ {
			for j := i + 1; j <= len(joined); j++ {
				subs = append(subs, joined[i:j])
			}
		}
		pr := make(map[bigram]int32)
		for _, a := range subs {
			for _, b := range subs {
				if rng.Intn(2) == 0 {
					pr[bigram{a, b}] = int32(rng.Intn(6))
				}
			}
		}
		want := naiveMergeSymbols(pr, syms)
		// Twice: with no vocabulary every pair is keyed by text; with every substring in it, by ids
		// (mergeTable), except the odd ones left out, which stay keyed by text.
		for _, withVocab := range []bool{false, true} {
			tok := &Tokenizer{}
			if withVocab {
				tok.vocab = map[string]int32{}
				for _, sub := range subs {
					if _, ok := tok.vocab[sub]; !ok && rng.Intn(8) != 0 {
						tok.vocab[sub] = int32(len(tok.idToPiece))
						tok.idToPiece = append(tok.idToPiece, sub)
					}
				}
			}
			for k, r := range pr {
				tok.merges.add(tok, k.left, k.right, r)
			}
			got := tok.mergeSymbols(append([]string(nil), syms...))
			if !slices.Equal(got, want) {
				t.Fatalf("trial %d (vocab %v): syms=%v\n got=%v\nwant=%v", trial, withVocab, syms, got, want)
			}
		}
	}
}

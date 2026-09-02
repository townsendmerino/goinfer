package tokenizer

import (
	"math/rand"
	"regexp"
	"strings"
	"testing"
	"unicode"
)

// AN INDEPENDENT ORACLE, NOT A SECOND COPY OF THE WALKER.
//
// splitO200k is hand-rolled because Go's RE2 cannot express `(?!\S)`. So the walker is checked
// against a matcher built the other way round: each alternative as its OWN anchored regexp, tried
// in pattern order at each position — which is exactly the ordered leftmost-match semantics HF's
// engine gives the full alternation. Only the one alternative RE2 cannot express is hand-handled,
// and it is the simplest of the seven.
//
// Writing the reference as "the same walk again" would have proved nothing; this shares no code
// with splitO200k and disagreeing with it is the whole point.
var o200kAlts = []*regexp.Regexp{
	regexp.MustCompile(`^[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`),
	regexp.MustCompile(`^[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`),
	regexp.MustCompile(`^\p{N}{1,3}`),
	regexp.MustCompile(`^ ?[^\s\p{L}\p{N}]+[\r\n/]*`),
	regexp.MustCompile(`^\s*[\r\n]+`),
	nil, // \s+(?!\S) — RE2 has no negative lookahead; handled below
	regexp.MustCompile(`^\s+`),
}

func refSplitO200k(s string) []string {
	var out []string
	for len(s) > 0 {
		matched := false
		for idx, re := range o200kAlts {
			var m string
			if re == nil {
				// \s+(?!\S), WITH BACKTRACKING — which is the whole subtlety and the reason this
				// alternative cannot be a plain prefix match. The engine takes the longest
				// whitespace run L >= 1 whose following position is end-of-input or another space.
				// Every interior position of a run IS a space, so when the run is followed by a
				// non-space the answer is run-1, not "no match": "  trailing" matches ONE space and
				// leaves the second to attach to the word. A first cut of this oracle only matched
				// a run at end-of-input, disagreed with the walker, and was itself the thing that
				// was wrong — the walker had it right, and so does splitGPT2, which shares the rule.
				ws := 0
				for _, r := range s {
					if !unicode.IsSpace(r) {
						break
					}
					ws += len(string(r))
				}
				switch {
				case ws == 0:
				case ws == len(s): // run reaches the end
					m = s[:ws]
				case ws >= 2: // leave the last space for the next token
					m = s[:ws-1]
				}
			} else if loc := re.FindStringIndex(s); loc != nil && loc[1] > 0 {
				m = s[:loc[1]]
			}
			if m != "" {
				out = append(out, m)
				s = s[len(m):]
				matched = true
				_ = idx
				break
			}
		}
		if !matched {
			r := []rune(s)[0]
			out = append(out, string(r))
			s = s[len(string(r)):]
		}
	}
	return out
}

// The four behaviours that differ from the cl100k walker, stated as cases so a regression names
// itself rather than showing up as an opaque diff.
func TestSplitO200k_theFourDifferencesFromCl100k(t *testing.T) {
	for _, tc := range []struct{ in, why string }{
		{"don't", "contractions ATTACH: one pre-token, not don + 't"},
		{"CamelCase", "a case transition splits"},
		{"2024", "digits cap at 3, so 202 + 4"},
		{"http://x", "the punctuation run swallows /"},
	} {
		got, want := splitO200k(tc.in), refSplitO200k(tc.in)
		if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
			t.Errorf("%q (%s):\n  walker %q\n  oracle %q", tc.in, tc.why, got, want)
		}
	}
	// And the headline one, spelled out: cl100k splits don't, o200k does not.
	if cl := splitGPT2("don't", 1); len(cl) < 2 {
		t.Fatalf("premise broke: the cl100k walker no longer splits don't (%q)", cl)
	}
	if o := splitO200k("don't"); len(o) != 1 {
		t.Errorf("o200k split don't into %q; the attached contraction is the difference", o)
	}
}

// Differential over targeted text plus random runes: the walker must agree with the oracle
// everywhere, not just on the four cases someone thought to write down.
func TestSplitO200k_matchesTheOrderedAlternationOracle(t *testing.T) {
	fixed := []string{
		"", " ", "\n", "\r\n", "  \n  ", "a", "A", "aA", "Aa", "ABCdef", "abcDEF",
		"What happened in 2024? I don't know.", "It's HTTPError vs HttpError",
		"naïve café", "e\u0301cole", // combining acute: \p{M} is word content
		"https://example.com/a/b", "x/y//z", "1234567", "  trailing   ",
		"新しい日本語のテキスト", "Ελληνικά ΚΕΦΑΛΑΙΑ", "don'T DON'T dOn'T",
		"a\n\nb", "\t\tx", "--flag=value", "'quoted'", "O'Neill's",
	}
	for _, s := range fixed {
		got, want := splitO200k(s), refSplitO200k(s)
		if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
			t.Errorf("%q:\n  walker %q\n  oracle %q", s, got, want)
		}
	}

	rng := rand.New(rand.NewSource(7))
	alphabet := []rune("aAzZ09 \n\r\t'/.,-_!?éÉ\u0301新Ω")
	for range 4000 {
		n := rng.Intn(24)
		b := make([]rune, n)
		for i := range b {
			b[i] = alphabet[rng.Intn(len(alphabet))]
		}
		s := string(b)
		got, want := splitO200k(s), refSplitO200k(s)
		if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
			t.Fatalf("random %q:\n  walker %q\n  oracle %q", s, got, want)
		}
	}
}

// Whatever it splits into, the pieces must reassemble to the input — a walker that drops or
// duplicates a rune would otherwise pass any spot check that happened to miss the position.
func TestSplitO200k_isLossless(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	alphabet := []rune("aA0 \n'/é新")
	for range 2000 {
		b := make([]rune, rng.Intn(20))
		for i := range b {
			b[i] = alphabet[rng.Intn(len(alphabet))]
		}
		s := string(b)
		if joined := strings.Join(splitO200k(s), ""); joined != s {
			t.Fatalf("not lossless: %q -> %q", s, joined)
		}
	}
}

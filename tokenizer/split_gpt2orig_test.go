package tokenizer

import (
	"math/rand"
	"regexp"
	"strings"
	"testing"
	"unicode"
)

// Independent oracle for GPT-2's own alternation, built the same way splitO200k's is: each
// alternative as its own anchored regexp, tried in pattern order, with only the lookahead
// alternative hand-handled. Shares no code with the walker.
var gpt2OrigAlts = []*regexp.Regexp{
	regexp.MustCompile(`^(?:'s|'t|'re|'ve|'m|'ll|'d)`), // case-SENSITIVE: GPT-2's regex has no (?i:)
	regexp.MustCompile(`^ ?\p{L}+`),
	regexp.MustCompile(`^ ?\p{N}+`),
	regexp.MustCompile(`^ ?[^\s\p{L}\p{N}]+`),
	nil, // \s+(?!\S)
	regexp.MustCompile(`^\s+`),
}

func refSplitGPT2Original(s string) []string {
	var out []string
	for len(s) > 0 {
		var m string
		for _, re := range gpt2OrigAlts {
			if re == nil {
				ws := 0
				for _, r := range s {
					if !unicode.IsSpace(r) {
						break
					}
					ws += len(string(r))
				}
				switch {
				case ws == 0:
				case ws == len(s):
					m = s[:ws]
				case ws >= 2:
					m = s[:ws-1]
				}
			} else if loc := re.FindStringIndex(s); loc != nil && loc[1] > 0 {
				m = s[:loc[1]]
			}
			if m != "" {
				break
			}
		}
		if m == "" {
			r := []rune(s)[0]
			m = string(r)
		}
		out = append(out, m)
		s = s[len(m):]
	}
	return out
}

// The headline difference: ` 2020` is ONE pre-token in GPT-2 and five under the cl100k walker with
// a 1-digit cap. That is the C-10 divergence for every GPT-2-family model this repo loads.
func TestSplitGPT2Original_digitsRunUnbounded(t *testing.T) {
	got := splitGPT2Original(" 2020")
	if len(got) != 1 || got[0] != " 2020" {
		t.Errorf("GPT-2 split %q into %q, want one pre-token", " 2020", got)
	}
	if cl := splitGPT2(" 2020", 1); len(cl) < 4 {
		t.Fatalf("premise broke: the cl100k walker no longer splits ' 2020' per digit (%q)", cl)
	}
}

func TestSplitGPT2Original_matchesTheOrderedAlternationOracle(t *testing.T) {
	fixed := []string{
		"", " ", "\n", "\r\n", " 2020", "don't", "DON'T", "It's", "It'S",
		"Hello, world!\n\nBye", "a  b", "  lead", "trail  ", "x/y//z",
		"CamelCase", "naïve café", "新しい", "12345 67", "-- flag=value",
	}
	for _, s := range fixed {
		got, want := splitGPT2Original(s), refSplitGPT2Original(s)
		if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
			t.Errorf("%q:\n  walker %q\n  oracle %q", s, got, want)
		}
	}
	rng := rand.New(rand.NewSource(13))
	alphabet := []rune("aAzZ09 \n\r\t'/.,-_!?éÉ新Ω")
	for range 4000 {
		b := make([]rune, rng.Intn(22))
		for i := range b {
			b[i] = alphabet[rng.Intn(len(alphabet))]
		}
		s := string(b)
		got, want := splitGPT2Original(s), refSplitGPT2Original(s)
		if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
			t.Fatalf("random %q:\n  walker %q\n  oracle %q", s, got, want)
		}
	}
}

func TestSplitGPT2Original_isLossless(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	alphabet := []rune("aA0 \n'/é")
	for range 2000 {
		b := make([]rune, rng.Intn(18))
		for i := range b {
			b[i] = alphabet[rng.Intn(len(alphabet))]
		}
		s := string(b)
		if joined := strings.Join(splitGPT2Original(s), ""); joined != s {
			t.Fatalf("not lossless: %q -> %q", s, joined)
		}
	}
}

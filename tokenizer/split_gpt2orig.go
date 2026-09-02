package tokenizer

import "unicode"

// splitGPT2Original walks GPT-2's OWN pre-tokenizer alternation:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// NOT the cl100k one splitGPT2 implements, despite the name that file carries. Two differences,
// both of which change ordinary text (audit-2026-09-02 C-10):
//
//   - Digits run UNBOUNDED with an optional leading space: ` 2020` is ONE pre-token here, where the
//     cl100k walker with a 1-digit cap gives `Ġ`,`2`,`0`,`2`,`0` — five.
//   - There are no `[\r\n]` clauses on the punctuation run, so a newline after punctuation starts a
//     fresh whitespace token instead of being swallowed.
//
// The contraction clause is case-SENSITIVE here (GPT-2's regex has no `(?i:)`), which the cl100k
// families do not share; `'S` is punctuation-plus-letter rather than a contraction.
//
// Validated by differential testing against an independent ordered-alternative matcher, the same
// way splitO200k is — see split_gpt2orig_test.go.
func splitGPT2Original(s string) []string {
	rs := []rune(s)
	n := len(rs)
	var out []string
	emit := func(a, b int) { out = append(out, string(rs[a:b])) }

	isPunct := func(r rune) bool {
		return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}

	for i := 0; i < n; {
		r := rs[i]

		// Alt 1: the contractions, CASE-SENSITIVE (no (?i:) in GPT-2's regex).
		if r == '\'' && i+1 < n {
			switch rs[i+1] {
			case 's', 't', 'm', 'd':
				emit(i, i+2)
				i += 2
				continue
			case 'r', 'v', 'l':
				if i+2 < n {
					c1, c2 := rs[i+1], rs[i+2]
					if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
						emit(i, i+3)
						i += 3
						continue
					}
				}
			}
		}

		// Alts 2-4: " ?" then a run of letters, numbers, or punctuation. The optional space is
		// greedy and gives itself back when what follows is not the alternative's class, which is
		// why each is tried with the space and then without it.
		matched := -1
		for _, class := range []func(rune) bool{unicode.IsLetter, unicode.IsNumber, isPunct} {
			for _, withSpace := range []bool{true, false} {
				j := i
				if withSpace {
					if r != ' ' || i+1 >= n {
						continue
					}
					j = i + 1
				}
				if j < n && class(rs[j]) {
					k := j
					for k < n && class(rs[k]) {
						k++
					}
					matched = k
					break
				}
			}
			if matched > 0 {
				break
			}
		}
		if matched > 0 {
			emit(i, matched)
			i = matched
			continue
		}

		// Alts 5-6: \s+(?!\S) then \s+ — the same backtracking rule the other walkers use: the
		// longest whitespace run whose next position is end-of-input or another space, which is
		// run-1 when a non-space follows.
		if unicode.IsSpace(r) {
			w := i
			for w < n && unicode.IsSpace(rs[w]) {
				w++
			}
			switch {
			case w == n:
				emit(i, w)
			case w-1 > i:
				emit(i, w-1)
				i = w - 1
				continue
			default:
				emit(i, w)
			}
			i = w
			continue
		}

		emit(i, i+1)
		i++
	}
	return out
}

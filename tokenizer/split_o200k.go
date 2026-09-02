package tokenizer

import "unicode"

// splitO200k walks the o200k / gpt-4o alternation:
//
//	[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//	[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//	\p{N}{1,3}
//	 ?[^\s\p{L}\p{N}]+[\r\n/]*
//	\s*[\r\n]+
//	\s+(?!\S)
//	\s+
//
// FOUR DIFFERENCES FROM splitGPT2, each of which changes real text (audit-2026-09-02 C-10):
//
//   - Contractions ATTACH to the word they follow instead of standing alone, so `don't` is ONE
//     pre-token where the cl100k walker emits `don` + `'t`.
//   - A case transition splits: the two word alternatives are (upper*)(lower+) and (upper+)(lower*),
//     so `CamelCase` is `Camel` + `Case`.
//   - Combining marks \p{M} are WORD CONTENT, so a decomposed accent stays with its letter instead
//     of falling into the punctuation run.
//   - The punctuation run's trailing class is [\r\n/]*, i.e. it swallows `/` — `http://` differs.
//
// Digits are capped at 3 by the pattern itself, so this takes no cap parameter; the cl100k walker
// needs one because that family's cap varies (Qwen 1, Llama-3 3).
//
// Written against the published o200k_base pattern above and validated by differential testing
// against an independent ordered-alternative matcher (split_o200k_test.go), not by eyeballing.
func splitO200k(s string) []string {
	rs := []rune(s)
	n := len(rs)
	var out []string
	emit := func(a, b int) { out = append(out, string(rs[a:b])) }

	isNL := func(r rune) bool { return r == '\r' || r == '\n' }
	// The word classes. Upper covers Lu/Lt/Lm/Lo/M; lower covers Ll/Lm/Lo/M. Lm, Lo and M are in
	// BOTH, which is what makes a script without case run as one word either way.
	isUpperish := func(r rune) bool {
		return unicode.IsUpper(r) || unicode.Is(unicode.Lt, r) || unicode.Is(unicode.Lm, r) ||
			unicode.Is(unicode.Lo, r) || unicode.Is(unicode.M, r)
	}
	isLowerish := func(r rune) bool {
		return unicode.IsLower(r) || unicode.Is(unicode.Lm, r) || unicode.Is(unicode.Lo, r) ||
			unicode.Is(unicode.M, r)
	}
	isPunct := func(r rune) bool {
		return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}
	// contractionLen returns the length of an attached contraction at i, or 0.
	contractionLen := func(i int) int {
		if i >= n || rs[i] != '\'' || i+1 >= n {
			return 0
		}
		switch unicode.ToLower(rs[i+1]) {
		case 's', 't', 'm', 'd':
			return 2
		case 'r', 'v', 'l':
			if i+2 < n {
				c1, c2 := unicode.ToLower(rs[i+1]), unicode.ToLower(rs[i+2])
				if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
					return 3
				}
			}
		}
		return 0
	}

	for i := 0; i < n; {
		r := rs[i]

		// Alts 1 and 2: an optional leading non-(CRLF/letter/number), then a word. Alt 1 is
		// upper* lower+ ; alt 2 is upper+ lower*. Trying them in order is what splits a case
		// transition: `CamelCase` matches alt 1 as `Camel`, leaving `Case`.
		//
		// THE OPTIONAL PREFIX BACKTRACKS. `X?` is greedy but the engine gives it up when the rest
		// of the alternative cannot match, so each alternative is tried WITH the prefix and then
		// WITHOUT it before moving on. A first cut committed to the prefix once and shared it
		// between both alternatives, which mis-split a combining mark followed by an uppercase
		// letter: it consumed the mark as the prefix, failed to find a lower run after `É`, and ran
		// on into alt 2 with the prefix already spent, emitting `◌́É` where the pattern gives
		// `◌́` then `É`. The differential oracle found it; no hand-written case would have.
		hasPrefix := !isNL(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
		// alt1 matches [Lu Lt Lm Lo M]* [Ll Lm Lo M]+ at j — and the two classes OVERLAP on
		// {Lm, Lo, M}, so the greedy star has to be able to give runes back. Take the longest
		// upper-ish run, then shrink it until the lower-ish run can match at least one, which is
		// what a backtracking engine does and is the first success in ITS order.
		//
		// A shortcut here (stopping the star at the first rune that is also lower-ish) was wrong on
		// exactly the overlap: a combining mark before uppercase, `◌́ΩÉéé`, matched only the mark
		// where the pattern matches the whole word. Found by the differential oracle, twice — this
		// is the second time the shortcut looked obviously right.
		alt1 := func(j int) int {
			kMax := j
			for kMax < n && isUpperish(rs[kMax]) {
				kMax++
			}
			for k := kMax; k >= j; k-- {
				lo := k
				for lo < n && isLowerish(rs[lo]) {
					lo++
				}
				if lo > k {
					return lo + contractionLen(lo)
				}
			}
			return -1
		}
		// alt2 matches [Lu Lt Lm Lo M]+ [Ll Lm Lo M]* at j. No backtracking is needed: the star is
		// allowed to be empty, so the greedy plus never has to give anything back.
		alt2 := func(j int) int {
			k := j
			for k < n && isUpperish(rs[k]) {
				k++
			}
			if k == j {
				return -1
			}
			lo := k
			for lo < n && isLowerish(rs[lo]) {
				lo++
			}
			return lo + contractionLen(lo)
		}
		matched := -1
		for _, try := range []struct {
			alt    func(int) int
			prefix bool
		}{{alt1, true}, {alt1, false}, {alt2, true}, {alt2, false}} {
			j := i
			if try.prefix {
				if !hasPrefix || i+1 >= n {
					continue
				}
				j = i + 1
			}
			if e := try.alt(j); e > 0 {
				matched = e
				break
			}
		}
		if matched > 0 {
			emit(i, matched)
			i = matched
			continue
		}

		// Alt 3: \p{N}{1,3}
		if unicode.IsNumber(r) {
			k := i + 1
			for k < n && k-i < 3 && unicode.IsNumber(rs[k]) {
				k++
			}
			emit(i, k)
			i = k
			continue
		}

		// Alt 4: " ?" [^\s\p{L}\p{N}]+ [\r\n/]* — note the '/' in the trailing class.
		{
			p := i
			if r == ' ' {
				p = i + 1
			}
			if p < n && isPunct(rs[p]) {
				k := p
				for k < n && isPunct(rs[k]) {
					k++
				}
				for k < n && (isNL(rs[k]) || rs[k] == '/') {
					k++
				}
				emit(i, k)
				i = k
				continue
			}
		}

		// Alts 5-7: whitespace runs, in priority order — identical to the cl100k walker's.
		if unicode.IsSpace(r) {
			w := i
			for w < n && unicode.IsSpace(rs[w]) {
				w++
			}
			last := -1
			for k := i; k < w; k++ {
				if isNL(rs[k]) {
					last = k
				}
			}
			if last >= 0 { // Alt 5: \s*[\r\n]+
				emit(i, last+1)
				i = last + 1
				continue
			}
			if w == n { // Alt 6: \s+(?!\S) at end of text
				emit(i, w)
			} else if w-1 > i { // all but the last space; it attaches to the next token
				emit(i, w-1)
				i = w - 1
				continue
			} else { // Alt 7: a lone space before a non-space
				emit(i, w)
			}
			i = w
			continue
		}

		// Nothing matched (a lone CR/LF-adjacent oddity): emit the rune so the walk terminates.
		emit(i, i+1)
		i++
	}
	return out
}

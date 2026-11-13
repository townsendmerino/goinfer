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
// FOUR DIFFERENCES FROM splitGPT2, each of which changes real text:
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
// Written against the published o200k_base pattern above and tested differentially against an
// independent ordered-alternative matcher (split_o200k_test.go).
func splitO200k(s string) []string { return splitO200kVariant(s, true, 3, false) }

// splitTekken walks Mistral's Tekken pattern (Voxtral, Mistral Small 3.x): o200k's alternation with the contraction suffixes REMOVED and `\p{N}` for a single digit:
//
//	[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Everything else (the case split, marks as word content, the `/` in the punctuation tail, the whitespace alternatives) is o200k's, so this is the same walker with two switches.
// `don't` is therefore `don` + `'t` here and one pre-token in o200k, and `2026` is four pre-tokens here and two in o200k.
func splitTekken(s string) []string { return splitO200kVariant(s, false, 1, false) }

// isFlintCJKKana reports whether r is in Flint's second Split class:
// [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff] (Han, Hiragana, Katakana).
func isFlintCJKKana(r rune) bool {
	return (r >= 0x4e00 && r <= 0x9fa5) || (r >= 0x3040 && r <= 0x309f) || (r >= 0x30a0 && r <= 0x30ff)
}

// splitFlint walks RecursiveSI/flint's three-stage Split pre-tokenizer sequence:
//
//  1. \p{N}{1,3} (Isolated)
//  2. [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+ (Isolated)
//  3. [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|
//     [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|
//     ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+ (Isolated)
//
// Because stages 1 and 2 isolate digit runs (≤3) and Han/Hiragana/Katakana runs before stage 3 runs,
// a leading space or punctuation before a digit or CJK/Kana run stays in the preceding span rather
// than attaching to it, and CJK/Kana runs split out of adjacent Latin words (`hello日本語world` ->
// `hello` + `日本語` + `world`).
func splitFlint(s string) []string {
	if s == "" {
		return nil
	}
	rs, off, s := runeOffsets(s)
	n := len(rs)
	out := make([]string, 0, n/3+1)
	spanStart := 0
	flushSpan := func(end int) {
		if end > spanStart {
			out = append(out, splitO200kVariant(s[off[spanStart]:off[end]], true, 0, true)...)
		}
	}
	for i := 0; i < n; {
		r := rs[i]
		if unicode.IsNumber(r) {
			flushSpan(i)
			k := i + 1
			for k < n && k-i < 3 && unicode.IsNumber(rs[k]) {
				k++
			}
			out = append(out, s[off[i]:off[k]])
			i = k
			spanStart = i
			continue
		}
		if isFlintCJKKana(r) {
			flushSpan(i)
			k := i + 1
			for k < n && isFlintCJKKana(rs[k]) {
				k++
			}
			out = append(out, s[off[i]:off[k]])
			i = k
			spanStart = i
			continue
		}
		i++
	}
	flushSpan(n)
	return out
}

// splitO200kVariant is the o200k walker with its family switches: whether contractions attach to the word they follow,
// the digit-run cap, and whether the punctuation clause is Flint's ` ?[\p{P}\p{S}]+[\r\n]*` instead of o200k's
// ` ?[^\s\p{L}\p{N}]+[\r\n/]*`.
func splitO200kVariant(s string, contractions bool, maxDigits int, flintPunct bool) []string {
	rs, off, s := runeOffsets(s)
	n := len(rs)
	out := make([]string, 0, n/3+1)
	gapStart := -1
	flushGap := func(pos int) {
		if gapStart >= 0 {
			out = append(out, s[off[gapStart]:off[pos]])
			gapStart = -1
		}
	}
	emit := func(a, b int) {
		flushGap(a)
		out = append(out, s[off[a]:off[b]])
	}

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
		if flintPunct {
			return unicode.IsPunct(r) || unicode.IsSymbol(r)
		}
		return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}
	// contractionLen returns the length of an attached contraction at i, or 0.
	contractionLen := func(i int) int {
		if !contractions || i >= n || rs[i] != '\'' || i+1 >= n {
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
		// WITHOUT it before moving on. Committing to the prefix once and sharing it between both
		// alternatives mis-splits a combining mark followed by an uppercase letter (`◌́É` where the
		// pattern gives `◌́` then `É`).
		hasPrefix := !isNL(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
		// alt1 matches [Lu Lt Lm Lo M]* [Ll Lm Lo M]+ at j. The two classes OVERLAP on {Lm, Lo, M}, so
		// the greedy star has to be able to give runes back: take the longest upper-ish run, then shrink
		// it until the lower-ish run can match at least one, as a backtracking engine does.
		//
		// Do not stop the star at the first rune that is also lower-ish: that is wrong on exactly the
		// overlap (a combining mark before uppercase, `◌́ΩÉéé`, matches only the mark where the pattern
		// matches the whole word).
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

		// Alt 3: \p{N}{1,3} (o200k) or \p{N} (Tekken): maxDigits
		if maxDigits > 0 && unicode.IsNumber(r) {
			k := i + 1
			for k < n && k-i < maxDigits && unicode.IsNumber(rs[k]) {
				k++
			}
			emit(i, k)
			i = k
			continue
		}

		// Alt 4: " ?" [^\s\p{L}\p{N}]+ [\r\n/]* (o200k/Tekken) or " ?" [\p{P}\p{S}]+ [\r\n]* (Flint).
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
				for k < n && (isNL(rs[k]) || (!flintPunct && rs[k] == '/')) {
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

		// Nothing matched (e.g. a Unicode category C rune under flintPunct's [\p{P}\p{S}] class):
		// accumulate into the Isolated unmatched gap until the next match or end of span.
		if gapStart < 0 {
			gapStart = i
		}
		i++
	}
	flushGap(n)
	return out
}

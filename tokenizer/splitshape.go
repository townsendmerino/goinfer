package tokenizer

import (
	"regexp"
	"strings"
)

// The pre-tokenizer SHAPE problem.
//
// splitGPT2 is exactly one regex: the cl100k / Llama-3 alternation, parameterised only by the
// digit-run cap. A tokenizer whose `Split` regex has a different shape, or a GGUF whose
// `tokenizer.ggml.pre` is not a known name, would be tokenized by the wrong walker silently: no error,
// no log, and `count_tokens` and usage drift by the same amount.
//
// This file does not guess a walker for an unknown shape. It NAMES the shape, so a mismatch can be
// reported (PreTokenizerDecline) instead of silently mis-tokenized.

// splitShape identifies which pre-tokenizer alternation a Split regex is.
type splitShape int

const (
	shapeUnknown splitShape = iota
	// shapeCl100k is the GPT-2/cl100k/Llama-3/Qwen alternation splitGPT2 implements: contractions
	// stand alone, letters group without case transitions, digits run up to the cap.
	shapeCl100k
	// shapeO200k is the gpt-4o / o200k family: contractions ATTACH to the preceding word, case
	// transitions split, combining marks are word content, and the punctuation run swallows `/`.
	shapeO200k
	// shapeGPT2Original is GPT-2's own
	// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
	// (see split_gpt2orig.go): a CASE-SENSITIVE contraction clause (no `(?i:)`, unlike cl100k's)
	// and no [\r\n] handling, so ` 2020` is ONE pre-token.
	shapeGPT2Original
	// shapeTekken is Mistral's Tekken pattern: o200k's alternation without the attached contractions and with a single `\p{N}` for digits (split_o200k.go's splitTekken).
	shapeTekken
)

func (s splitShape) String() string {
	switch s {
	case shapeCl100k:
		return "cl100k"
	case shapeO200k:
		return "o200k"
	case shapeGPT2Original:
		return "gpt2-original"
	case shapeTekken:
		return "tekken"
	}
	return "unknown"
}

// canonSplit strips the incidental differences between two spellings of the same alternation:
// whitespace outside character classes, and the `(?i:…)` vs `(?i)` spelling of the contraction
// group. It is deliberately conservative — it normalises FORM, never meaning.
func canonSplit(re string) string {
	r := strings.ReplaceAll(re, "\n", "")
	r = strings.ReplaceAll(r, "\t", "")
	return r
}

var (
	// The o200k markers, each necessary and none of them present in the cl100k shape.
	o200kCaseSplit = regexp.MustCompile(`\[\\p\{Lu\}\\p\{Lt\}\\p\{Lm\}\\p\{Lo\}\\p\{M\}\]`)
	o200kAttached  = regexp.MustCompile(`\+\(\?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d\)\?`)
	// The cl100k contraction clause stands alone at the head of the alternation, wrapped
	// case-INSENSITIVE.
	cl100kLeadContraction = regexp.MustCompile(`^\(\?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d\)\|`)
	// GPT-2's own: a ` ?\p{L}+` letters clause.
	gpt2Letters = regexp.MustCompile(` \?\\p\{L\}\+`)
	// GPT-2's own contraction clause stands alone at the head too, but UNWRAPPED and case-SENSITIVE: no
	// `(?i:)`. It is a positive marker, because the real GPT-2 regex (split_gpt2orig.go) does carry a
	// contraction clause, so classifying a Split spelling it by exclusion would call it shapeUnknown.
	gpt2LeadContraction = regexp.MustCompile(`^'s\|'t\|'re\|'ve\|'m\|'ll\|'d\|`)
	// Tekken: the o200k case-split classes, NO attached contractions, then a bare `\p{N}` alternative followed by o200k's punctuation tail.
	tekkenDigitTail = regexp.MustCompile(`\|\\p\{N\}\| \?\[\^\\s\\p\{L\}\\p\{N\}\]\+\[\\r\\n/\]\*`)
)

// classifySplit names the alternation a Split regex expresses.
//
// It answers "which walker implements this", not "is this valid" — an unrecognised shape is
// shapeUnknown, which the caller reports rather than silently walking with the wrong one.
func classifySplit(re string) splitShape {
	c := canonSplit(re)
	if c == "" {
		return shapeUnknown
	}
	switch {
	case o200kCaseSplit.MatchString(c) && o200kAttached.MatchString(c):
		return shapeO200k
	case o200kCaseSplit.MatchString(c) && !o200kAttached.MatchString(c) && tekkenDigitTail.MatchString(c):
		return shapeTekken
	case cl100kLeadContraction.MatchString(c):
		return shapeCl100k
	case gpt2Letters.MatchString(c) && gpt2LeadContraction.MatchString(c):
		return shapeGPT2Original
	}
	return shapeUnknown
}

// walkerImplements reports whether this build has a walker for the shape.
//
// shapeUnknown never does, by definition: an unclassified regex keeps the cl100k walker and sets
// PreTokenizerDecline, because a wrong split that SAYS SO is better than one that does not, and
// refusing to load a model that works today would be the worse trade.
func walkerImplements(s splitShape) bool {
	return s == shapeCl100k || s == shapeO200k || s == shapeGPT2Original || s == shapeTekken
}

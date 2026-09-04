package tokenizer

import (
	"regexp"
	"strings"
)

// The pre-tokenizer SHAPE problem (audit-2026-09-02 C-10).
//
// splitGPT2 is exactly one regex: the cl100k / Llama-3 alternation, parameterised only by the
// digit-run cap. Nothing ever compared a tokenizer's ACTUAL `Split` regex against that shape, and a
// GGUF's `tokenizer.ggml.pre` outside a four-name switch fell to a default that is still that
// walker. So a family whose pre-tokenizer differs was tokenized by the wrong one, silently: no
// error, no log, and `count_tokens` and usage drift by the same amount.
//
// Measured on this machine's own assets, which is what turned the audit's "medium confidence on the
// exact pre strings" into a fact:
//
//	gpt-oss-20b-MXFP4.gguf       pre="gpt-4o"   -> not in the switch -> default
//	Qwen3.5-35B-A3B-Q4_K_M.gguf  pre="qwen35"   -> not in the switch -> default
//	Qwen3 / qwen2.5-coder        pre="qwen2"    -> recognised
//
// Both unrecognised ones are families this repo ships AND gates. The gpt-oss case is the audit's;
// the qwen35 case is not in the audit, and it is the quieter of the two — "qwen35" falls to a
// default that differs from the "qwen2" branch only in NFC being OFF, so it diverges on exactly the
// inputs that need normalising and on nothing else.
//
// This file does not guess a walker for an unknown shape. It NAMES the shape, so a mismatch can be
// reported instead of silently mis-tokenized.

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
)

func (s splitShape) String() string {
	switch s {
	case shapeCl100k:
		return "cl100k"
	case shapeO200k:
		return "o200k"
	case shapeGPT2Original:
		return "gpt2-original"
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
	// GPT-2's own contraction clause stands alone at the head too, but UNWRAPPED and
	// case-SENSITIVE — no `(?i:)`. This is the positive marker V-15 (docs/review-2026-09-04.md)
	// added: the real GPT-2 regex (split_gpt2orig.go's own docstring, transcribed from OpenAI's
	// source) DOES carry a contraction clause, so a Split spelling it was misclassified as
	// shapeUnknown by an exclusion that assumed the opposite.
	gpt2LeadContraction = regexp.MustCompile(`^'s\|'t\|'re\|'ve\|'m\|'ll\|'d\|`)
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
	return s == shapeCl100k || s == shapeO200k || s == shapeGPT2Original
}

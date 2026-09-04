package tokenizer

import (
	"errors"
	"io/fs"
	"os"
	"testing"
)

// The three real alternations, verbatim. The cl100k and o200k ones are what
// ~/models/qwen3-0.6b-bf16 and the o200k family ship; the GPT-2 one is GPT-2's own.
const (
	reCl100kQwen  = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	reCl100kLlama = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	reO200k       = `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	reGPT2        = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
)

// C-10: splitGPT2 implements ONE alternation and was applied to every family. Nothing compared a
// tokenizer's actual Split regex against it, so a family whose pre-tokenizer differs was walked by
// the wrong one — no error, no log, and count_tokens and usage drift with it.
//
// Naming the shape is what makes a mismatch reportable. This pins the classifier against the real
// regexes rather than against a paraphrase of them.
func TestClassifySplit_namesTheRealAlternations(t *testing.T) {
	for name, tc := range map[string]struct {
		re   string
		want splitShape
	}{
		"qwen (cl100k, 1 digit)":   {reCl100kQwen, shapeCl100k},
		"llama3 (cl100k, 3 digit)": {reCl100kLlama, shapeCl100k},
		"o200k / gpt-4o":           {reO200k, shapeO200k},
		"gpt-2 original":           {reGPT2, shapeGPT2Original},
		"empty":                    {"", shapeUnknown},
		"something else":           {`\w+|\s+`, shapeUnknown},
	} {
		if got := classifySplit(tc.re); got != tc.want {
			t.Errorf("%s: classified as %v, want %v", name, got, tc.want)
		}
	}
}

// THE POINT OF NAMING THE SHAPE: a shape claimed as implemented must HAVE a walker. If this ever
// reports true for one that does not, the mis-tokenization is back and silent again — which is the
// whole of C-10.
func TestWalkerImplements_onlyShapesWithAWalker(t *testing.T) {
	for _, s := range []splitShape{shapeCl100k, shapeO200k, shapeGPT2Original} {
		if !walkerImplements(s) {
			t.Errorf("%v has a walker (splitGPT2 / splitO200k / splitGPT2Original) and is not claimed", s)
		}
	}
	// An unclassified regex still has none, and claiming it would re-hide exactly the divergence
	// this gate exists for.
	for _, s := range []splitShape{shapeUnknown} {
		if walkerImplements(s) {
			t.Errorf("%v is claimed as implemented and no walker exists for it", s)
		}
	}
}

// The dispatch must actually USE the o200k walker for an o200k tokenizer — a walker nothing selects
// is the shape of every call-site gap this audit has turned up so far.
func TestSplitPre_dispatchesOnTheDeclaredShape(t *testing.T) {
	o := &Tokenizer{preShape: shapeO200k, maxDigits: 1}
	if got := o.splitPre("don't"); len(got) != 1 {
		t.Errorf("an o200k tokenizer split don't into %q; the walker was not selected", got)
	}
	c := &Tokenizer{preShape: shapeCl100k, maxDigits: 1}
	if got := c.splitPre("don't"); len(got) != 2 {
		t.Errorf("a cl100k tokenizer produced %q for don't; want the contraction split off", got)
	}
	// An unknown shape keeps the cl100k walker rather than failing to tokenize at all.
	u := &Tokenizer{preShape: shapeUnknown, maxDigits: 1}
	if got := u.splitPre("don't"); len(got) != 2 {
		t.Errorf("an unknown shape produced %q; it must fall back to the cl100k walker", got)
	}
	// GPT-2's own alternation: ` 2020` is one pre-token, which is the divergence that motivated it.
	g := &Tokenizer{preShape: shapeGPT2Original, maxDigits: 1}
	if got := g.splitPre(" 2020"); len(got) != 1 {
		t.Errorf("a gpt2-original tokenizer split ' 2020' into %q; the walker was not selected", got)
	}
}

// The o200k markers are what distinguish it, so a regex carrying only one of them must not be
// mistaken for it — that would route a family to a walker built for a different pattern.
func TestClassifySplit_partialO200kIsNotO200k(t *testing.T) {
	caseOnly := `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}]+|\p{N}{1,3}|\s+`
	if got := classifySplit(caseOnly); got == shapeO200k {
		t.Error("a regex with the case-transition class but no attached contractions classified as " +
			"o200k; the attached contraction is the half that changes don't from two pre-tokens to one")
	}
}

// C-10, the GGUF half: an unrecognised `tokenizer.ggml.pre` fell SILENTLY to GPT-2-like defaults.
//
// The two that do it are not hypothetical — measured on this machine's own assets:
//
//	gpt-oss-20b-MXFP4.gguf       pre="gpt-4o"
//	Qwen3.5-35B-A3B-Q4_K_M.gguf  pre="qwen35"
//
// Both are families this repo ships and gates. qwen35 is now recognised (same pipeline as qwen2,
// and the default differed from it only in NFC being OFF — a divergence on exactly the inputs that
// need normalising, which is why nobody saw it). gpt-4o is NOT recognised, because recognising it
// would mean claiming the walker handles o200k, and it does not.
func TestByteLevelKnobs_unknownPreIsReportedNotAssumed(t *testing.T) {
	for _, pre := range []string{"llama-bpe", "llama3", "llama-v3", "qwen2", "qwen2.5", "qwen",
		"qwen35", "mellum2", "gpt-2", "default", "", "gpt-4o"} {
		if _, _, _, _, _, _, known := byteLevelKnobs(pre); !known {
			t.Errorf("pre=%q is not recognised; it was, before this change or after it", pre)
		}
	}
	// The real unrecognised names, from the assets and from the audit's list.
	for _, pre := range []string{"llama4", "kimi-k2", "deepseek-v3", "glm4", "something-new"} {
		if _, _, _, _, _, _, known := byteLevelKnobs(pre); known {
			t.Errorf("pre=%q reports as known; the switch does not carry it, so the knobs are the "+
				"cl100k default and the caller must be TOLD rather than served silently", pre)
		}
	}

	// qwen35 specifically: it must match qwen2 and NOT the default, which differs in NFC.
	d2, f2, on2, im2, sd2, _, _ := byteLevelKnobs("qwen2")
	d35, f35, on35, im35, sd35, _, _ := byteLevelKnobs("qwen35")
	if d2 != d35 || f2 != f35 || on2 != on35 || im2 != im35 || sd2 != sd35 {
		t.Errorf("qwen35 knobs (%v,%v,%v,%v,%v) differ from qwen2's (%v,%v,%v,%v,%v)",
			d35, f35, on35, im35, sd35, d2, f2, on2, im2, sd2)
	}
	if _, _, onDefault, _, _, _, _ := byteLevelKnobs("unrecognised"); onDefault == on35 {
		t.Error("the premise broke: the default's NFC setting now equals qwen35's, so falling " +
			"through would have been harmless and this test no longer describes the defect")
	}
}

// gpt-oss: the GGUF says pre="gpt-4o" and carries no regex, so the mapping to a walker has to be
// justified from outside the file. It is: the SAME MODEL's HF tokenizer.json declares a Split regex
// byte-identical to reO200k below — sha256 2d1b8dc1… on both, checked against
// ~/models/gpt-oss-20b-hf. That is evidence, not the name-to-shape guess C-10 is about.
func TestByteLevelKnobs_gptOssSelectsTheO200kWalker(t *testing.T) {
	digits, _, normOn, _, _, shape, known := byteLevelKnobs("gpt-4o")
	if !known {
		t.Fatal("pre=gpt-4o is not recognised")
	}
	if shape != shapeO200k {
		t.Errorf("pre=gpt-4o selects %v, want o200k", shape)
	}
	if digits != 3 {
		t.Errorf("digit cap %d, want 3 — the o200k pattern caps at \\p{N}{1,3}", digits)
	}
	if normOn {
		t.Error("NFC is on; gpt-oss's tokenizer.json ships no normalizer")
	}
	// And the regex this repo pins for o200k must still classify as o200k — if the constant and
	// the classifier ever disagree, the mapping above is resting on nothing.
	if classifySplit(reO200k) != shapeO200k {
		t.Error("reO200k no longer classifies as o200k")
	}
}

// TestClassifySplit_realGPT2PatternIsNotUnknown pins V-15 (docs/review-2026-09-04.md): the real
// GPT-2 regex (split_gpt2orig.go's own docstring, transcribed from OpenAI's source) DOES carry a
// contraction clause — just case-sensitive and unwrapped, unlike cl100k's `(?i:...)` one.
// classifySplit's shapeGPT2Original case used to require the clause's ABSENCE
// (`!strings.Contains(c, "'s|'t|'re")`), so a Split spelling the actual pattern classified as
// shapeUnknown and silently fell back to the cl100k walker instead — never reaching the walker
// (splitGPT2Original) written specifically for it.
func TestClassifySplit_realGPT2PatternIsNotUnknown(t *testing.T) {
	const real = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
	if got := classifySplit(real); got != shapeGPT2Original {
		t.Errorf("classifySplit(the real GPT-2 pattern) = %v, want shapeGPT2Original (V-15) — "+
			"it was misclassified as shapeUnknown by an exclusion that assumed the clause absent",
			got)
	}
	// Regression guard in the other direction: cl100k's case-INSENSITIVE, wrapped spelling of the
	// same six contractions must still classify as cl100k, not gpt2-original — the discriminator
	// is the (?i:) wrapping, not merely whether a contraction clause is present at all.
	if got := classifySplit(reCl100kQwen); got != shapeCl100k {
		t.Errorf("classifySplit(cl100k) = %v, want shapeCl100k — the fix must not blur the two "+
			"contraction spellings together", got)
	}
}

// TestInitByteLevel_bareByteLevelUseRegexIsGPT2Original pins V-15's second half: a real HF `gpt2`
// export's pre_tokenizer is a bare (non-Sequence) `{"type":"ByteLevel","use_regex":true}` with no
// separate Split node at all (testdata/gpt2/onnx/tokenizer.json — verified by reading the file,
// not assumed). splitRegex returns "" for it, the same empty result Mellum2's genuinely
// regex-agnostic Digits+ByteLevel Sequence produces — but here "empty" means "use my built-in
// GPT-2 regex" (HF's own semantics for use_regex:true), not "no opinion". The old code treated
// both the same way and silently kept the cl100k walker with no PreTokenizerDecline at all.
func TestInitByteLevel_bareByteLevelUseRegexIsGPT2Original(t *testing.T) {
	const dir = "../testdata/gpt2/onnx"
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("skip: %s absent", dir)
	}
	tk, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s): %v", dir, err)
	}
	if tk.mode != modeByteLevel {
		t.Fatalf("gpt2 fixture loaded as mode %d, want modeByteLevel", tk.mode)
	}
	if tk.preShape != shapeGPT2Original {
		t.Errorf("preShape = %v, want shapeGPT2Original (V-15) — a bare ByteLevel{use_regex:true} "+
			"with no Split node was not recognised as GPT-2's own regex", tk.preShape)
	}
	if got := tk.PreTokenizerDecline(); got != "" {
		t.Errorf("PreTokenizerDecline = %q, want \"\" — this build DOES have a walker "+
			"(splitGPT2Original) for this shape", got)
	}
	// THE END-TO-END CHECK: ` 2020` must walk as one pre-token (GPT-2's own unbounded digit run),
	// not five (the cl100k walker's 1-digit cap) — the exact divergence this finding is about.
	if got := tk.splitPre(" 2020"); len(got) != 1 {
		t.Errorf("splitPre(\" 2020\") = %q (%d pre-tokens), want 1 — the cl100k walker was used "+
			"instead of GPT-2's own (V-15)", got, len(got))
	}
}

// A Sequence-wrapped pre_tokenizer (Mellum2: Digits + ByteLevel) must NOT be mistaken for the
// bare-ByteLevel case — its top-level `type` is "Sequence", not "ByteLevel", so
// isBareByteLevelUseRegex must read false for it, or Mellum2 would be silently reclassified as a
// side effect of a fix aimed at plain gpt2.
func TestIsBareByteLevelUseRegex_sequenceWrappedIsNotBare(t *testing.T) {
	const sequenceWrapped = `{"type":"Sequence","pretokenizers":[` +
		`{"type":"Digits","individual_digits":true},` +
		`{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}`
	if isBareByteLevelUseRegex([]byte(sequenceWrapped)) {
		t.Error("a Sequence-wrapped ByteLevel{use_regex:true} (Mellum2's real shape) read as bare")
	}
	const bare = `{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}`
	if !isBareByteLevelUseRegex([]byte(bare)) {
		t.Error("a top-level ByteLevel{use_regex:true} (real gpt2's shape) did not read as bare")
	}
}

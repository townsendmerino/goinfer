package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// Byte-level BPE — the GPT-2 / Llama-3 / Qwen family (G3). Same ordered-merge
// core as Gemma (mergeSymbols); the wrapper differs: NFC normalize, a GPT-2
// split-regex pretokenizer, and a byte→printable-rune map so every initial
// symbol is in-vocab (no byte-fallback).

// initByteLevel builds the byte↔unicode tables and resolves special tokens
// from tokenizer_config.json. Unlike Gemma, no special token is required —
// bos is commonly null (these families add no BOS at encode time), so a
// missing token resolves to -1 ("none") rather than a load error.
func (t *Tokenizer) initByteLevel(tj *tokenizerJSON, dir string) error {
	t.byteEncoder, t.byteDecoder = buildByteLevelTables()
	t.normForm, t.normOn = normalizerForm(tj.Normalizer)
	t.maxDigits = digitRunCap(splitRegex(tj.PreTokenizer))
	t.splitDigits = hasIndividualDigits(tj.PreTokenizer)

	t.special = SpecialTokens{BOS: -1, EOS: -1, Pad: -1, StartOfTurn: -1, EndOfTurn: -1}
	lookup := func(piece string) int {
		if id, ok := t.vocab[piece]; piece != "" && ok {
			return int(id)
		}
		return -1
	}
	cfg := readTokenizerConfig(dir) // best-effort; missing file → empty
	t.special.BOS = lookup(cfg.BosToken)
	t.special.EOS = lookup(cfg.EosToken)
	t.special.Pad = lookup(cfg.PadToken)
	// Chat turn markers, for the demo's chat template — the byte-level
	// families standardize on <|im_start|>/<|im_end|>. Left -1 if absent.
	t.special.StartOfTurn = lookup("<|im_start|>")
	t.special.EndOfTurn = lookup("<|im_end|>")
	t.chatTemplate = cfg.ChatTemplate // for chat.Detect; "" if absent
	return nil
}

// buildByteLevelTables reproduces GPT-2's bytes_to_unicode: the printable
// byte ranges map to themselves; every other byte is assigned a fresh
// codepoint from U+0100 upward, in ascending byte order. So 0x20 (space) →
// U+0120 'Ġ', 0x0A (newline) → U+010A 'Ċ', 0x09 (tab) → U+0109 'ĉ'. Every one
// of the 256 runes is its own vocab entry, which is why byte-level BPE needs
// no byte-fallback.
func buildByteLevelTables() ([256]rune, map[rune]byte) {
	var enc [256]rune
	dec := make(map[rune]byte, 256)
	printable := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	n := 0
	for b := range 256 {
		r := rune(b)
		if !printable(b) {
			r = rune(256 + n)
			n++
		}
		enc[b] = r
		dec[r] = byte(b)
	}
	return enc, dec
}

// encodeByteLevel is the byte-level analogue of Encode: split out added/special
// tokens on the raw text (longest match), NFC-normalize each gap, pretokenize
// with the GPT-2 regex, then byte-level-BPE each pretoken.
func (t *Tokenizer) encodeByteLevel(text string, addBOS bool) ([]int, error) {
	var out []int32
	if addBOS && t.special.BOS >= 0 {
		out = append(out, int32(t.special.BOS))
	}

	gapStart := 0
	emitGap := func(end int) error {
		if end <= gapStart {
			return nil
		}
		gap := text[gapStart:end]
		if t.normOn {
			gap = t.normForm.String(gap)
		}
		// A Digits{individual_digits} pretokenizer (Mellum2) runs before the
		// byte-level regex: it isolates each digit into its own segment first, so
		// the GPT-2 regex never sees a digit adjacent to a leading space (" 1"
		// becomes " " + "1", not the single "Ġ1" piece). Without it the gap is one
		// segment.
		for _, seg := range t.digitSegments(gap) {
			for _, piece := range splitGPT2(seg, t.maxDigits) {
				ids, err := t.bpeByteLevel(piece)
				if err != nil {
					return err
				}
				out = append(out, ids...)
			}
		}
		return nil
	}

	i := 0
	for i < len(text) {
		if id, n := t.added.match(text, i); n > 0 {
			if err := emitGap(i); err != nil {
				return nil, err
			}
			out = append(out, id)
			i += n
			gapStart = i
			continue
		}
		_, sz := utf8.DecodeRuneInString(text[i:])
		if sz == 0 {
			sz = 1 // defensive: never stall on invalid UTF-8
		}
		i += sz
	}
	if err := emitGap(len(text)); err != nil {
		return nil, err
	}

	res := make([]int, len(out))
	for k, v := range out {
		res[k] = int(v)
	}
	return res, nil
}

// bpeByteLevel maps a single pretoken to ids: byte-level-encode it to a string
// of printable runes, honor ignore_merges (a whole pretoken that is itself a
// vocab entry wins before any merging), then run the shared merge loop over
// per-rune symbols.
func (t *Tokenizer) bpeByteLevel(piece string) ([]int32, error) {
	if piece == "" {
		return nil, nil
	}
	var enc strings.Builder
	for _, b := range []byte(piece) {
		enc.WriteRune(t.byteEncoder[b])
	}
	s := enc.String()

	if t.ignoreMerges {
		if id, ok := t.vocab[s]; ok {
			return []int32{id}, nil
		}
	}

	syms := make([]string, 0, len(s))
	for _, r := range s {
		syms = append(syms, string(r))
	}
	syms = t.mergeSymbols(syms)

	ids := make([]int32, len(syms))
	for k, sym := range syms {
		id, ok := t.vocab[sym]
		if !ok {
			// Unreachable for a well-formed byte-level tokenizer: the 256 base
			// runes and every merge result are vocab entries. Surface it
			// rather than emit a wrong id.
			return nil, fmt.Errorf("tokenizer: byte-level symbol %q not in vocab", sym)
		}
		ids[k] = id
	}
	return ids, nil
}

// decodeByteLevel inverts the byte-level map: render each piece (special tokens
// included — their surface chars are all printable bytes that map to
// themselves), translate every rune back to its byte, and interpret the byte
// stream as UTF-8.
func (t *Tokenizer) decodeByteLevel(ids []int) (string, error) {
	var buf []byte
	for _, id := range ids {
		if id < 0 || id >= len(t.idToPiece) {
			return "", fmt.Errorf("tokenizer.Decode: id %d out of range [0,%d)", id, len(t.idToPiece))
		}
		for _, r := range t.idToPiece[id] {
			if b, ok := t.byteDecoder[r]; ok {
				buf = append(buf, b)
			} else {
				buf = utf8.AppendRune(buf, r) // defensive: non-byte-level rune
			}
		}
	}
	return string(buf), nil
}

// splitGPT2 reproduces the Qwen/Llama-3 pretokenizer split — the GPT-2 regex
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
//	 ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Go's regexp (RE2) can't express the `(?!\S)` lookahead, so we walk the runes
// applying the alternatives in priority order, each greedy — the same ordered,
// leftmost-match semantics HF's regex engine uses. Returns the raw substrings
// (before byte-level encoding).
//
// maxDigits is the digit-run cap: the `\p{N}` alternative — one digit for Qwen,
// `\p{N}{1,3}` (runs of up to three) for Llama-3 — read from the tokenizer's
// pretokenizer regex.
func splitGPT2(s string, maxDigits int) []string {
	rs := []rune(s)
	n := len(rs)
	var out []string
	emit := func(a, b int) { out = append(out, string(rs[a:b])) }

	isNL := func(r rune) bool { return r == '\r' || r == '\n' }
	// The [^\s\p{L}\p{N}] class: not whitespace, letter, or number.
	isPunct := func(r rune) bool {
		return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}

	for i := 0; i < n; {
		r := rs[i]

		// Alt 1: contractions (case-insensitive), ASCII apostrophe only.
		if r == '\'' && i+1 < n {
			switch unicode.ToLower(rs[i+1]) {
			case 's', 't', 'm', 'd':
				emit(i, i+2)
				i += 2
				continue
			case 'r', 'v', 'l':
				if i+2 < n {
					c1, c2 := unicode.ToLower(rs[i+1]), unicode.ToLower(rs[i+2])
					if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
						emit(i, i+3)
						i += 3
						continue
					}
				}
			}
		}

		// Alt 2: [^\r\n\p{L}\p{N}]? \p{L}+ — an optional leading non-(CRLF/
		// letter/number) char, then one or more letters (this is how a leading
		// space attaches to the following word).
		{
			j := i
			if !isNL(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r) && i+1 < n && unicode.IsLetter(rs[i+1]) {
				j = i + 1
			}
			if unicode.IsLetter(rs[j]) {
				k := j
				for k < n && unicode.IsLetter(rs[k]) {
					k++
				}
				emit(i, k)
				i = k
				continue
			}
		}

		// Alt 3: \p{N}{1,maxDigits} — a run of up to maxDigits number runes
		// (one per token for Qwen, runs of ≤3 for Llama-3).
		if unicode.IsNumber(r) {
			k := i + 1
			for k < n && k-i < maxDigits && unicode.IsNumber(rs[k]) {
				k++
			}
			emit(i, k)
			i = k
			continue
		}

		// Alt 4: " ?" [^\s\p{L}\p{N}]+ [\r\n]* — an optional single ASCII
		// space, a punctuation/symbol run, then any trailing newlines.
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
				for k < n && isNL(rs[k]) {
					k++
				}
				emit(i, k)
				i = k
				continue
			}
		}

		// Alts 5–7 cover whitespace runs, in priority order.
		if unicode.IsSpace(r) {
			w := i
			for w < n && unicode.IsSpace(rs[w]) {
				w++
			}
			// Alt 5: \s*[\r\n]+ — end the run at its last newline.
			last := -1
			for k := i; k < w; k++ {
				if isNL(rs[k]) {
					last = k
				}
			}
			if last >= 0 {
				emit(i, last+1)
				i = last + 1
				continue
			}
			// Alt 6: \s+(?!\S) — a run at end-of-text, or all but the last
			// space of a run that precedes a non-space (that last space is
			// left to attach to the next token via Alt 2/4's optional prefix).
			if w == n {
				emit(i, w)
			} else if w-1 > i {
				emit(i, w-1)
				i = w - 1
				continue
			} else {
				// Alt 7: \s+ — a lone space before a non-space.
				emit(i, w)
			}
			i = w
			continue
		}

		// Defensive: unreachable for valid input, but never stall.
		emit(i, i+1)
		i++
	}
	return out
}

// normalizerForm reads tokenizer.json's normalizer and returns the Unicode
// normalization form to apply, if any. Qwen normalizes NFC; Llama-3 has no
// normalizer (null). A Sequence normalizer is scanned for the first form.
func normalizerForm(raw json.RawMessage) (norm.Form, bool) {
	if len(raw) == 0 {
		return norm.NFC, false
	}
	var node struct {
		Type        string            `json:"type"`
		Normalizers []json.RawMessage `json:"normalizers"`
	}
	if json.Unmarshal(raw, &node) != nil {
		return norm.NFC, false
	}
	switch node.Type {
	case "NFC":
		return norm.NFC, true
	case "NFKC":
		return norm.NFKC, true
	case "NFD":
		return norm.NFD, true
	case "NFKD":
		return norm.NFKD, true
	case "Sequence":
		for _, sub := range node.Normalizers {
			if f, ok := normalizerForm(sub); ok {
				return f, true
			}
		}
	}
	return norm.NFC, false
}

// splitRegex extracts the pretokenizer split pattern from tokenizer.json's
// pre_tokenizer, which is either a Split node directly or a Sequence whose
// first Split carries it. Empty when absent.
func splitRegex(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var node struct {
		Pattern struct {
			Regex string `json:"Regex"`
		} `json:"pattern"`
		Pretokenizers []json.RawMessage `json:"pretokenizers"`
	}
	if json.Unmarshal(raw, &node) != nil {
		return ""
	}
	if node.Pattern.Regex != "" {
		return node.Pattern.Regex
	}
	for _, sub := range node.Pretokenizers {
		if r := splitRegex(sub); r != "" {
			return r
		}
	}
	return ""
}

// digitSegments splits s the way a Digits{individual_digits:true} pretokenizer
// does, when one is present (t.splitDigits): each numeric rune becomes its own
// segment and maximal non-numeric runs are their own segments, so a later
// byte-level regex never merges a digit with an adjacent space or letter. Without
// such a pretokenizer it returns s unchanged (one segment). Uses the same
// unicode.IsNumber predicate as splitGPT2's \p{N} alternative.
func (t *Tokenizer) digitSegments(s string) []string {
	if !t.splitDigits || s == "" {
		return []string{s}
	}
	var segs []string
	runStart, inRun := 0, false // inRun: accumulating a non-digit run
	flush := func(end int) {
		if end > runStart {
			segs = append(segs, s[runStart:end])
		}
	}
	for i, r := range s {
		if unicode.IsNumber(r) {
			flush(i) // close any pending non-digit run
			segs = append(segs, string(r))
			runStart, inRun = i+len(string(r)), false
		} else if !inRun {
			runStart, inRun = i, true
		}
	}
	flush(len(s))
	return segs
}

// hasIndividualDigits reports whether tokenizer.json's pre_tokenizer contains a
// Digits node with individual_digits set (Mellum2's pipeline), recursing into a
// Sequence's pretokenizers.
func hasIndividualDigits(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var node struct {
		Type            string            `json:"type"`
		IndividualDigit bool              `json:"individual_digits"`
		Pretokenizers   []json.RawMessage `json:"pretokenizers"`
	}
	if json.Unmarshal(raw, &node) != nil {
		return false
	}
	if node.Type == "Digits" && node.IndividualDigit {
		return true
	}
	return slices.ContainsFunc(node.Pretokenizers, hasIndividualDigits)
}

var digitRunRe = regexp.MustCompile(`\\p\{N\}\{1,(\d+)\}`)

// digitRunCap reads the pretokenizer's digit-run cap from its regex: Llama-3's
// `\p{N}{1,3}` groups up to three digits, Qwen's bare `\p{N}` takes one. Any
// pattern without an explicit `{1,N}` digit clause caps at one.
func digitRunCap(pattern string) int {
	if m := digitRunRe.FindStringSubmatch(pattern); m != nil {
		if n, err := strconv.Atoi(m[1]); err == nil && n >= 1 {
			return n
		}
	}
	return 1
}

// tokenizerConfig is the subset of tokenizer_config.json we read for special
// tokens (the byte-level families don't hardcode them in tokenizer.json).
type tokenizerConfig struct {
	BosToken     string
	EosToken     string
	PadToken     string
	ChatTemplate string
}

func readTokenizerConfig(dir string) tokenizerConfig {
	var cfg tokenizerConfig
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	if err != nil {
		return cfg
	}
	var m struct {
		BosToken     json.RawMessage `json:"bos_token"`
		EosToken     json.RawMessage `json:"eos_token"`
		PadToken     json.RawMessage `json:"pad_token"`
		ChatTemplate json.RawMessage `json:"chat_template"`
	}
	if json.Unmarshal(raw, &m) != nil {
		return cfg
	}
	cfg.BosToken = tokenContent(m.BosToken)
	cfg.EosToken = tokenContent(m.EosToken)
	cfg.PadToken = tokenContent(m.PadToken)
	// chat_template is usually a JSON string; ignore the rarer list form.
	_ = json.Unmarshal(m.ChatTemplate, &cfg.ChatTemplate)
	return cfg
}

// tokenContent extracts the surface form from an HF token-config value, which
// is either a JSON string ("<|im_end|>"), the AddedToken object form
// {"content": "..."}, or null.
func tokenContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var obj struct {
		Content string `json:"content"`
	}
	if json.Unmarshal(raw, &obj) == nil {
		return obj.Content
	}
	return ""
}

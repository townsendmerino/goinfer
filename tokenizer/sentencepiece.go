package tokenizer

import (
	"container/heap"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// spaceMarker is SentencePiece's "▁" (U+2581 LOWER ONE EIGHTH BLOCK), which
// Gemma's normalizer substitutes for every ASCII space before BPE.
const spaceMarker = "▁"

// maxTokenID bounds a token id read from a tokenizer.json: idToPiece is sized to
// maxID+1, so an unbounded id would overflow (int32) or OOM. ~2M is an order of
// magnitude above the largest real vocabulary (~300K).
const maxTokenID = 1 << 21

// SpecialTokens holds the Gemma chat/control token ids resolved from the
// vocab at load time. The generation loop needs BOS/EOS; the chat template
// needs the turn markers.
type SpecialTokens struct {
	BOS         int // <bos>
	EOS         int // <eos>
	Pad         int // <pad>
	StartOfTurn int // <start_of_turn>
	EndOfTurn   int // <end_of_turn>
}

// bigram is a comparable map key for the ordered BPE merge table: the pair
// (left, right) → its merge rank (lower = higher priority, merged first).
type bigram struct{ left, right string }

// tokMode selects the pre/post-processing pipeline wrapped around the shared
// ordered-merge BPE core. The merge table is reused verbatim across families;
// only how text becomes initial symbols (and ids become text) differs.
type tokMode int

const (
	// modeGemma is Gemma 3's SentencePiece-style byte-fallback BPE: normalize
	// ASCII space → ▁, no pretokenizer, per-rune symbols with <0xNN> fallback
	// for out-of-vocab runes. (M2.)
	modeGemma tokMode = iota
	// modeByteLevel is the GPT-2 / Llama-3 / Qwen byte-level BPE: NFC
	// normalize, a GPT-2 split-regex pretokenizer, then map each UTF-8 *byte*
	// to a printable rune (space → Ġ) so every symbol is in-vocab — no
	// byte-fallback. (G3.)
	modeByteLevel
)

// Tokenizer is a loaded BPE tokenizer. It serves two families behind one
// merge core (see tokMode): Gemma 3's byte-fallback SentencePiece-style model
// (M2) and the byte-level GPT-2/Llama-3/Qwen model (G3). Load reads the HF
// tokenizer.json and resolves the mode + special tokens from it.
//
// The pipeline mirrors HF `tokenizers` exactly so ids match the per-family
// golden: split out added/special tokens (longest match on the raw text),
// normalize the gaps, pretokenize (byte-level only), then BPE each piece.
type Tokenizer struct {
	// preDecline is non-empty when the pre-tokenizer this tokenizer declares is one this build
	// does not walk. See PreTokenizerDecline.
	preDecline string
	// preShape is the alternation the model's own Split regex expresses; it selects the walker.
	preShape splitShape

	vocab     map[string]int32 // piece → id
	idToPiece []string         // id → piece (len == vocab size)
	pairRank  map[bigram]int32 // BPE merge rank
	special   SpecialTokens

	// scoreRank is the SentencePiece unigram fallback (non-nil ⇒ used instead of pairRank in the
	// merge loop): id → merge priority, densely ranked by descending token score. Some SPM exports
	// ship tokenizer.ggml.scores but NOT tokenizer.ggml.merges (gemma-3 GGUFs), so there is no BPE
	// merge list. llama.cpp then encodes by greedily merging the adjacent pair whose CONCATENATION
	// is the highest-scoring vocab token — this rank reproduces that order (score DESC, id ASC on a
	// tie) so the shared min-heap merge core pops the same pair. See mergeRank / buildScoreRank.
	scoreRank []int32

	mode tokMode

	// modeGemma: byte-fallback.
	byteFallback bool
	bytePiece    [256]string // b → "<0xNN>" piece (the byte-fallback tokens)
	byteToVal    map[int32]byte
	unkPiece     string
	unkID        int32
	// SentencePiece dummy-prefix knobs. Llama-2/Mistral SPM prepend a ▁ to each
	// normalized gap on encode and strip one leading space on decode; Gemma 3
	// does neither. Detected from the normalizer (tokenizer.json) or set by the
	// GGUF loader (tokenizer.ggml.model == "llama").
	prependSpace      bool
	stripLeadingSpace bool

	// modeByteLevel: byte↔unicode map + the whole-piece-wins flag, plus the
	// two pipeline knobs that vary across byte-level families (read from
	// tokenizer.json): the normalizer and the pretokenizer's digit-run cap.
	ignoreMerges bool
	byteEncoder  [256]rune     // byte → printable rune (GPT-2 bytes_to_unicode)
	byteDecoder  map[rune]byte // rune → byte (inverse)
	maxDigits    int           // pretokenizer digit-run cap: Qwen \p{N}=1, Llama-3 \p{N}{1,3}=3
	normForm     norm.Form     // Unicode normalization form (when normOn)
	normOn       bool          // Qwen normalizes NFC; Llama-3 has no normalizer
	splitDigits  bool          // a Digits{individual_digits} pretokenizer runs before the byte-level regex (Mellum2): isolate each digit, so a leading space never attaches to one

	added *addedTrie // added/special token surface forms → id

	chatTemplate string // raw GGUF/HF tokenizer.chat_template (Jinja); "" if absent. For chat.Detect.
}

// ChatTemplate returns the model's raw chat-template string (GGUF
// tokenizer.chat_template / HF tokenizer_config.json chat_template), or "" if
// the checkpoint carries none. The chat package fingerprints it to pick a
// native renderer.
func (t *Tokenizer) ChatTemplate() string { return t.chatTemplate }

// Has reports whether the surface form exists as a token in the vocab — the
// probe chat.Detect uses to recognize a bare checkpoint with no chat template.
func (t *Tokenizer) Has(piece string) bool { _, ok := t.vocab[piece]; return ok }

// TokenID returns the id of a vocab piece (e.g. a chat stop marker like
// "<end_of_turn>") and whether it exists.
func (t *Tokenizer) TokenID(piece string) (int, bool) { id, ok := t.vocab[piece]; return int(id), ok }

// PreTokenizerDecline reports why this tokenizer's pre-tokenizer is NOT the alternation the walker
// implements, or "" when it is (or when the family is not byte-level at all).
//
// It exists because the answer used to be nothing. splitGPT2 is exactly the cl100k/Llama-3
// alternation, and every byte-level family was walked with it: a `Split` regex of a different shape,
// or a GGUF `tokenizer.ggml.pre` outside a four-name switch, produced a DIFFERENT id stream from
// HF and from llama.cpp with no error and no log — `count_tokens` and usage drifting by the same
// amount (audit-2026-09-02 C-10). Measured on this machine's own assets: gpt-oss's GGUF is
// pre="gpt-4o" and Qwen3.5's is pre="qwen35", and neither was in the switch.
//
// A caller that cares — serve's startup line, a gate — can now say so out loud. Nothing here
// changes what the walker does: naming the divergence is separable from fixing it, and shipping a
// walker for a pattern this repo cannot yet check against a reference would be the worse half.
func (t *Tokenizer) PreTokenizerDecline() string { return t.preDecline }

// Special returns the resolved special-token ids.
func (t *Tokenizer) Special() SpecialTokens { return t.special }

// Chat-template selection moved to the chat package: chat.Detect fingerprints
// ChatTemplate() (or falls back to Has() vocab markers) and returns a native
// renderer. (The old ChatStyle enum/heuristic lived here.)

// --- tokenizer.json schema (only the fields we need) ---

type tokenizerJSON struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
	Model struct {
		Type         string           `json:"type"`
		ByteFallback bool             `json:"byte_fallback"`
		IgnoreMerges bool             `json:"ignore_merges"`
		UnkToken     *string          `json:"unk_token"`
		Vocab        map[string]int32 `json:"vocab"`
		// Merges has two HF encodings: the newer pair-array form
		// [["a","b"],…] (Qwen3) and the older flat space-joined form
		// ["a b",…] (Llama-3, GPT-2). Kept raw and normalized by parseMerges.
		Merges json.RawMessage `json:"merges"`
	} `json:"model"`
	// Decoder.Type selects the pipeline family: "ByteLevel" → modeByteLevel
	// (GPT-2/Qwen/Llama-3), anything else → modeGemma (SentencePiece-style).
	Decoder struct {
		Type string `json:"type"`
	} `json:"decoder"`
	// Normalizer + PreTokenizer drive the two byte-level knobs that vary by
	// family (NFC-or-none, digit-run cap); kept raw and parsed in initByteLevel.
	Normalizer   json.RawMessage `json:"normalizer"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
}

// parseMerges normalizes the two HF merge encodings into [left,right] pairs.
// Newer files use a pair array ([["a","b"],…]); older ones (Llama-3, GPT-2)
// use one space-joined string per merge ("a b"). Byte-level pieces never
// contain a literal space (it is encoded as Ġ), so the single separating
// space is unambiguous.
func parseMerges(raw json.RawMessage) ([][2]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	// Pair-array form.
	var pairs [][]string
	if err := json.Unmarshal(raw, &pairs); err == nil {
		out := make([][2]string, len(pairs))
		for i, p := range pairs {
			if len(p) != 2 {
				return nil, fmt.Errorf("merge %d has %d parts, want 2", i, len(p))
			}
			out[i] = [2]string{p[0], p[1]}
		}
		return out, nil
	}
	// Flat space-joined form.
	var flat []string
	if err := json.Unmarshal(raw, &flat); err != nil {
		return nil, fmt.Errorf("merges: not a pair array or string array: %w", err)
	}
	out := make([][2]string, len(flat))
	for i, m := range flat {
		l, r, ok := strings.Cut(m, " ")
		if !ok {
			return nil, fmt.Errorf("merge %d %q has no space separator", i, m)
		}
		out[i] = [2]string{l, r}
	}
	return out, nil
}

// Load reads a SentencePiece/BPE model. path may point directly at a
// tokenizer.json or at a directory containing one (e.g. the HF checkpoint
// dir). The legacy tokenizer.model (SP protobuf) is not supported — the HF
// tokenizer.json carries the same vocab plus the explicit merge table and
// pipeline, which is what we match for parity.
func Load(path string) (*Tokenizer, error) {
	jsonPath := path
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		jsonPath = filepath.Join(path, "tokenizer.json")
	}
	raw, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, fmt.Errorf("tokenizer.Load: %w", err)
	}
	return parseTokenizerJSON(raw, jsonPath, filepath.Dir(jsonPath))
}

// LoadJSONBytes parses a tokenizer from raw tokenizer.json bytes — the dir-less twin of
// Load. Used to load the tokenizer carried in a prequant .giw built from a SAFETENSORS
// model, whose tok half is the tokenizer.json itself (not a GGUF metadata blob, which is
// what LoadGGUFBytes expects). Self-contained tokenizer.json only: it passes an EMPTY
// sibling dir so a byte-level pipeline never reads tokenizer_config.json — before this a
// bare "tokenizer.json" resolved siblings relative to the process CWD, silently adopting an
// unrelated tokenizer_config.json (wrong BOS at prefill, wrong template fingerprint) if one
// sat in the server's working directory (audit M-14).
func LoadJSONBytes(raw []byte) (*Tokenizer, error) {
	return parseTokenizerJSON(raw, "tokenizer.json", "")
}

// parseTokenizerJSON builds a Tokenizer from raw tokenizer.json bytes. jsonPath is the
// display path (for error messages only). siblingDir is where a byte-level pipeline looks
// for sibling files (tokenizer_config.json) — pass "" for no-sibling mode (a self-contained
// blob load), NOT filepath.Dir of a bare filename, which would resolve to "." (the CWD) and
// read an unrelated config (audit M-14). Split out from Load so the parse — the untrusted-input
// surface — is testable and fuzzable without a file on disk.
func parseTokenizerJSON(raw []byte, jsonPath, siblingDir string) (*Tokenizer, error) {
	var tj tokenizerJSON
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("tokenizer.Load: parse %s: %w", jsonPath, err)
	}
	// model.type is "BPE" for Gemma/Qwen3/Llama-3; GPT-2's tokenizer.json omits
	// it, but its merges + byte-level pipeline are the same BPE machinery, so an
	// empty type is accepted (anything else is a genuine mismatch).
	if tj.Model.Type != "BPE" && tj.Model.Type != "" {
		return nil, fmt.Errorf("tokenizer.Load: unsupported model type %q (want BPE)", tj.Model.Type)
	}
	if len(tj.Model.Vocab) == 0 {
		return nil, fmt.Errorf("tokenizer.Load: empty vocab in %s", jsonPath)
	}

	merges, err := parseMerges(tj.Model.Merges)
	if err != nil {
		return nil, fmt.Errorf("tokenizer.Load: parse %s: %w", jsonPath, err)
	}

	t := &Tokenizer{
		vocab:        tj.Model.Vocab,
		pairRank:     make(map[bigram]int32, len(merges)),
		byteFallback: tj.Model.ByteFallback,
		ignoreMerges: tj.Model.IgnoreMerges,
		byteToVal:    make(map[int32]byte, 256),
	}
	if tj.Decoder.Type == "ByteLevel" {
		t.mode = modeByteLevel
	}

	// id → piece, sized to the max id so every produced id is renderable.
	// Added/special tokens may live outside model.vocab with higher ids (Qwen
	// keeps its <|im_*|> tokens only in added_tokens), so fold them in too —
	// otherwise Decode can't render an emitted special id.
	// Every id must be a valid non-negative index below a sane ceiling. An
	// untrusted tokenizer.json could otherwise carry a negative id (index panic
	// in the fill loops below) or a huge one (maxID+1 overflows int32 to negative
	// → make panics, or a multi-GB idToPiece). The ceiling is far above any real
	// vocabulary (largest ~300K) but bounds the allocation.
	maxID := int32(-1)
	for piece, id := range tj.Model.Vocab {
		if id < 0 || id > maxTokenID {
			return nil, fmt.Errorf("tokenizer.Load: vocab id %d for %q out of range [0,%d]", id, piece, maxTokenID)
		}
		if id > maxID {
			maxID = id
		}
	}
	for _, a := range tj.AddedTokens {
		if a.ID < 0 || a.ID > maxTokenID {
			return nil, fmt.Errorf("tokenizer.Load: added-token id %d out of range [0,%d]", a.ID, maxTokenID)
		}
		if a.ID > maxID {
			maxID = a.ID
		}
	}
	t.idToPiece = make([]string, maxID+1)
	for piece, id := range tj.Model.Vocab {
		t.idToPiece[id] = piece
	}
	for _, a := range tj.AddedTokens {
		t.idToPiece[a.ID] = a.Content
		if _, ok := t.vocab[a.Content]; !ok {
			t.vocab[a.Content] = a.ID
		}
	}

	// Merge ranks: position in the list is the priority.
	for i, m := range merges {
		t.pairRank[bigram{m[0], m[1]}] = int32(i)
	}

	// Per-family setup: byte tables + special-token resolution differ.
	switch t.mode {
	case modeGemma:
		if err := t.initGemma(&tj); err != nil {
			return nil, err
		}
	case modeByteLevel:
		if err := t.initByteLevel(&tj, siblingDir); err != nil {
			return nil, err
		}
	}

	// Added-vocabulary trie: every added-token surface form is matched
	// (longest-first) against the raw text before normalization/BPE.
	t.added = newAddedTrie()
	for _, a := range tj.AddedTokens {
		t.added.add(a.Content, a.ID)
	}

	return t, nil
}

// initGemma sets up the Gemma 3 byte-fallback path: the "<0xNN>" byte tokens
// and the (required) Gemma special tokens. These are mandatory for this
// family, so a missing one is a load error — the M2 golden depends on them.
func (t *Tokenizer) initGemma(tj *tokenizerJSON) error {
	// SentencePiece dummy prefix: Llama-2/Mistral prepend a ▁ (and strip one
	// leading space on decode); Gemma 3 has no Prepend normalizer.
	if prependMarker(tj.Normalizer) {
		t.prependSpace = true
		t.stripLeadingSpace = true
	}

	// Byte-fallback tokens: "<0x00>".."<0xFF>".
	for b := range 256 {
		p := fmt.Sprintf("<0x%02X>", b)
		t.bytePiece[b] = p
		if id, ok := t.vocab[p]; ok {
			t.byteToVal[id] = byte(b)
		} else if t.byteFallback {
			return fmt.Errorf("tokenizer.Load: byte_fallback set but %q missing from vocab", p)
		}
	}

	t.unkPiece = "<unk>"
	if tj.Model.UnkToken != nil {
		t.unkPiece = *tj.Model.UnkToken
	}
	mustID := func(piece string) (int32, error) {
		id, ok := t.vocab[piece]
		if !ok {
			return 0, fmt.Errorf("tokenizer.Load: required token %q not in vocab", piece)
		}
		return id, nil
	}
	var err error
	if t.unkID, err = mustID(t.unkPiece); err != nil {
		return err
	}
	for _, r := range []struct {
		piece string
		dst   *int
	}{
		{"<bos>", &t.special.BOS}, {"<eos>", &t.special.EOS}, {"<pad>", &t.special.Pad},
	} {
		id, err := mustID(r.piece)
		if err != nil {
			return err
		}
		*r.dst = int(id)
	}
	// Chat turn markers are OPTIONAL and family-specific — Gemma 3 uses
	// <start_of_turn>/<end_of_turn>, Gemma 4 renamed them to <|turn>/<turn|>, Qwen uses
	// <|im_start|>/<|im_end|>. A plain-text/base checkpoint may have none. -1 = absent
	// (only chat-template rendering needs them; tokenization/generation doesn't). Mirrors
	// the gguf/bytelevel tokenizers, which already default these to -1.
	t.special.StartOfTurn, t.special.EndOfTurn = -1, -1
	for _, r := range []struct {
		pieces []string
		dst    *int
	}{
		{[]string{"<start_of_turn>", "<|turn>", "<|im_start|>"}, &t.special.StartOfTurn},
		{[]string{"<end_of_turn>", "<turn|>", "<|im_end|>"}, &t.special.EndOfTurn},
	} {
		for _, p := range r.pieces {
			if id, ok := t.vocab[p]; ok {
				*r.dst = int(id)
				break
			}
		}
	}
	return nil
}

// Segment is a span of a rendered chat prompt tagged with whether the tokenizer
// may recognize special/added-token surface forms inside it. Chat templates emit
// their structural markers as Special segments and untrusted message/tool content
// as non-special ones, so EncodeSegments can refuse to promote a "<|im_end|>" typed
// by a user into a real turn-boundary control token (M25 — the parse_special
// distinction). See EncodeSegments.
type Segment struct {
	Text    string
	Special bool
}

// Encode turns text into token ids. If addBOS, prepend the BOS token (the
// generation prefill expects it for Gemma; byte-level families with no BOS
// ignore the flag). Added/special tokens written literally in the text are
// recognized and emitted as their own ids — do NOT use this on untrusted content
// (a user message, a tool result); use EncodeSegments so injected marker strings
// stay literal (M25).
func (t *Tokenizer) Encode(text string, addBOS bool) ([]int, error) {
	return t.encode(text, addBOS, true)
}

// encode is Encode with an explicit parseSpecial: when false the added-token trie
// is not consulted, so the whole text BPEs as a single gap and any special surface
// form in it stays ordinary text. The result is otherwise identical to Encode —
// trusted gap text never contains a special form, so on legitimate input the two
// modes agree (the byte-identity property EncodeSegments relies on).
func (t *Tokenizer) encode(text string, addBOS, parseSpecial bool) ([]int, error) {
	// A vocab loaded without merges AND without scores can DECODE but not encode — refuse rather than
	// return a silently unmerged (wrong) tokenization. A SentencePiece vocab with scores but no merge
	// list (gemma-3 GGUFs) encodes via the score-rank path (mergeRank), so it is allowed. See gguf.go.
	if len(t.pairRank) == 0 && t.scoreRank == nil {
		return nil, fmt.Errorf("tokenizer: this vocab was loaded without merge ranks or scores (decode-only); cannot Encode")
	}
	if t.vocab == nil {
		return nil, fmt.Errorf("tokenizer.Encode: %w", errors.New("tokenizer not loaded"))
	}
	if t.mode == modeByteLevel {
		return t.encodeByteLevel(text, addBOS, parseSpecial)
	}
	var out []int32
	if addBOS && t.special.BOS >= 0 { // a GGUF llama-family model may carry no BOS key ⇒ BOS == -1; don't emit a garbage id (M28)
		out = append(out, int32(t.special.BOS))
	}

	gapStart := 0
	i := 0
	flushGap := func(end int) {
		if end > gapStart {
			out = append(out, t.bpe(t.normalizeGap(text[gapStart:end]))...)
		}
	}
	for parseSpecial && i < len(text) {
		if id, n := t.added.match(text, i); n > 0 {
			flushGap(i)
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
	flushGap(len(text))

	res := make([]int, len(out))
	for k, v := range out {
		res[k] = int(v)
	}
	return res, nil
}

// EncodeSegments tokenizes a rendered chat prompt from its Render segments: a
// Special segment is parsed WITH the added-token trie (its structural markers
// become control ids), a content segment WITHOUT it (a user/tool "<|im_end|>" stays
// literal text) — the standard parse_special split that stops prompt injection
// from forging turn boundaries (M25). addBOS prepends BOS once up front; templates
// that emit their own BOS marker pass addBOS=false. On legitimate input the id
// stream is identical to Encode(Render(...)): every content segment is exactly one
// gap between the template's genuine special tokens, so no cross-boundary merge is
// lost.
func (t *Tokenizer) EncodeSegments(segs []Segment, addBOS bool) ([]int, error) {
	var out []int
	if addBOS && t.special.BOS >= 0 {
		out = append(out, int(t.special.BOS))
	}
	for _, s := range segs {
		ids, err := t.encode(s.Text, false, s.Special)
		if err != nil {
			return nil, err
		}
		out = append(out, ids...)
	}
	return out, nil
}

// normalize applies the SentencePiece space normalizer: replace every ASCII
// space with the ▁ marker. (Tabs, newlines and other whitespace are left as-is
// — the added-vocabulary split handles the newline-run tokens.)
func normalize(s string) string {
	return strings.ReplaceAll(s, " ", spaceMarker)
}

// normalizeGap normalizes one raw text gap for the modeGemma BPE: replace
// spaces with ▁ and, for SentencePiece models that use a dummy prefix
// (prependSpace — Llama-2/Mistral), prepend a leading ▁. Gemma sets neither
// flag, so this is a plain space-replace there. The prepended ▁ is a literal
// marker, not a space, so the order relative to normalize is immaterial.
func (t *Tokenizer) normalizeGap(s string) string {
	s = normalize(s)
	if t.prependSpace {
		s = spaceMarker + s
	}
	return s
}

// prependMarker reports whether a normalizer prepends the ▁ SentencePiece
// dummy prefix (Llama-2/Mistral SPM do; Gemma 3 does not). Recurses a Sequence.
func prependMarker(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var node struct {
		Type        string            `json:"type"`
		Prepend     string            `json:"prepend"`
		Normalizers []json.RawMessage `json:"normalizers"`
	}
	if json.Unmarshal(raw, &node) != nil {
		return false
	}
	if node.Type == "Prepend" {
		return node.Prepend == spaceMarker
	}
	if node.Type == "Sequence" {
		if slices.ContainsFunc(node.Normalizers, prependMarker) {
			return true
		}
	}
	return false
}

// bpe segments a normalized gap into ids: per-rune initial symbols (byte
// fallback for out-of-vocab runes), then greedy merge of the lowest-rank
// adjacent pair (leftmost on ties) until none remain.
func (t *Tokenizer) bpe(gap string) []int32 {
	if gap == "" {
		return nil
	}
	syms := make([]string, 0, len(gap))
	for _, r := range gap {
		s := string(r)
		if _, ok := t.vocab[s]; ok {
			syms = append(syms, s)
			continue
		}
		if t.byteFallback {
			for _, b := range []byte(s) {
				syms = append(syms, t.bytePiece[b])
			}
			continue
		}
		syms = append(syms, t.unkPiece)
	}

	syms = t.mergeSymbols(syms)

	ids := make([]int32, len(syms))
	for k, s := range syms {
		if id, ok := t.vocab[s]; ok {
			ids[k] = id
		} else {
			ids[k] = t.unkID // unreachable under byte fallback
		}
	}
	return ids
}

// mergeSymbols is the shared BPE core: repeatedly merge the lowest-rank
// adjacent pair (leftmost on ties) until no adjacent pair has a known rank.
// Both families call it; only the initial symbol construction (per-rune +
// byte-fallback vs byte-level) and the id mapping around it differ. The merge
// table itself is identical HF data, so the merge loop is too.
// mergeSymbols applies BPE merges to syms in ascending rank order, leftmost first
// on a tie. Behaviorally identical to the naive "rescan for the globally best pair
// each step" (the golden-parity tests gate this), but O(n log n) via a min-heap over
// a doubly-linked list instead of O(n²): Gemma has no pretokenizer, so a whole
// inter-added-token gap arrives here as ONE unit, and the old rescan turned a few
// hundred KB of client text into minutes of CPU (M28).
func (t *Tokenizer) mergeSymbols(syms []string) []string {
	n := len(syms)
	if n < 2 {
		return syms
	}
	text := make([]string, n) // node text (grows as pairs merge into the left node)
	prev := make([]int32, n)
	next := make([]int32, n)
	alive := make([]bool, n)
	copy(text, syms)
	for i := range text {
		prev[i] = int32(i - 1)
		next[i] = int32(i + 1)
		alive[i] = true
	}
	next[n-1] = -1

	// A candidate's key packs (rank, leftNodeIndex) so the heap pops the lowest rank
	// and, on a tie, the leftmost node — the same choice the rescan made. Node indices
	// never change, and surviving nodes keep left-to-right order, so leftmost-by-index
	// == leftmost-in-the-old-array.
	h := &mergeHeap{}
	pushPair := func(left int32) {
		if left < 0 {
			return
		}
		r := next[left]
		if r < 0 {
			return
		}
		if rank, ok := t.mergeRank(text[left], text[r]); ok {
			heap.Push(h, mergeCand{key: int64(rank)<<32 | int64(left), left: left, right: r})
		}
	}
	for i := int32(0); i+1 < int32(n); i++ {
		pushPair(i)
	}
	for h.Len() > 0 {
		c := heap.Pop(h).(mergeCand)
		if !alive[c.left] || !alive[c.right] || next[c.left] != c.right {
			continue // one endpoint was already merged away
		}
		// The left node's text may have grown since this candidate was queued (a fresh
		// candidate with the correct rank was pushed then); re-derive the rank and drop
		// this one if it no longer matches.
		if rank, ok := t.mergeRank(text[c.left], text[c.right]); !ok || int64(rank)<<32|int64(c.left) != c.key {
			continue
		}
		text[c.left] += text[c.right]
		alive[c.right] = false
		next[c.left] = next[c.right]
		if nr := next[c.right]; nr >= 0 {
			prev[nr] = c.left
		}
		pushPair(prev[c.left]) // the pair to the left now has a new right text
		pushPair(c.left)       // and this node has a new right neighbor
	}
	out := syms[:0] // node 0 is never a right child, so it stays the live head
	for i := int32(0); i >= 0; i = next[i] {
		out = append(out, text[i])
	}
	return out
}

// mergeRank returns the merge priority of the adjacent pair (left, right): lower rank merges first,
// leftmost on a tie (the heap packs rank<<32|leftIndex). In BPE mode it is the pairRank of the two
// source pieces; in SPM-scores mode (scoreRank != nil, no merge list) it is the score-rank of the
// CONCATENATED token left+right, so the highest-scoring merge fires first — the SentencePiece order.
func (t *Tokenizer) mergeRank(left, right string) (int32, bool) {
	if t.scoreRank != nil {
		if id, ok := t.vocab[left+right]; ok {
			return t.scoreRank[id], true
		}
		return 0, false
	}
	r, ok := t.pairRank[bigram{left, right}]
	return r, ok
}

// buildScoreRank turns per-id SentencePiece scores into a merge-priority rank: rank 0 is the
// highest-scoring token, so a lower rank merges first (matching mergeSymbols' min-heap). Tokens with
// the SAME score share a rank, so on a score tie the heap key's leftIndex — not the token id —
// decides, and two equal-score merges at different positions fire left-to-right, matching llama.cpp's
// leftmost-position SPM order. (Previously every id got a distinct rank via a lower-id tiebreak, so a
// same-score merge on a lower id fired ahead of a leftward one — a silent divergence from the
// reference tokenization on a vocab with equal-score competing merges: audit R-29.)
func buildScoreRank(scores []float32) []int32 {
	order := make([]int32, len(scores))
	for i := range order {
		order[i] = int32(i)
	}
	sort.Slice(order, func(a, b int) bool {
		if scores[order[a]] != scores[order[b]] {
			return scores[order[a]] > scores[order[b]]
		}
		return order[a] < order[b] // stable, deterministic within an equal-score run
	})
	rank := make([]int32, len(scores))
	for r, id := range order {
		if r > 0 && scores[id] == scores[order[r-1]] {
			rank[id] = rank[order[r-1]] // same score → same rank (leftmost position breaks the tie)
		} else {
			rank[id] = int32(r)
		}
	}
	return rank
}

// mergeCand is one candidate BPE merge on the heap; key = rank<<32 | leftIndex.
type mergeCand struct {
	key         int64
	left, right int32
}

type mergeHeap []mergeCand

func (h mergeHeap) Len() int           { return len(h) }
func (h mergeHeap) Less(i, j int) bool { return h[i].key < h[j].key }
func (h mergeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *mergeHeap) Push(x any)        { *h = append(*h, x.(mergeCand)) }
func (h *mergeHeap) Pop() any {
	old := *h
	n := len(old)
	c := old[n-1]
	*h = old[:n-1]
	return c
}

// Decode turns token ids back into text: render each piece (with ▁ → space),
// fusing runs of byte-fallback pieces back into their raw UTF-8 bytes. Special
// tokens render as their literal surface form (e.g. "<eos>") — the generation
// loop is responsible for stopping at EOS, not Decode.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	// Whole-sequence decode DOES strip the single leading dummy-prefix space
	// (Llama-2/Mistral; Gemma leaves it) — it is the prefix of the SEQUENCE.
	return t.decode(ids, t.stripLeadingSpace)
}

// decode renders ids to text; stripLeading applies the SentencePiece dummy-prefix
// leading-space strip to the assembled string. The strip is a sequence-level concern
// (the space belongs to the first token of the whole sequence), so per-piece callers
// pass false — see DecodePiece.
func (t *Tokenizer) decode(ids []int, stripLeading bool) (string, error) {
	if t.mode == modeByteLevel {
		return t.decodeByteLevel(ids)
	}
	var sb strings.Builder
	var pending []byte
	flush := func() {
		if len(pending) > 0 {
			sb.Write(pending)
			pending = pending[:0]
		}
	}
	for _, id := range ids {
		if id < 0 || id >= len(t.idToPiece) {
			return "", fmt.Errorf("tokenizer.Decode: id %d out of range [0,%d)", id, len(t.idToPiece))
		}
		if b, ok := t.byteToVal[int32(id)]; ok {
			pending = append(pending, b)
			continue
		}
		flush()
		sb.WriteString(strings.ReplaceAll(t.idToPiece[id], spaceMarker, " "))
	}
	flush()
	out := sb.String()
	if stripLeading {
		out = strings.TrimPrefix(out, " ")
	}
	return out, nil
}

// DecodeContinuation decodes ids that CONTINUE an existing sequence rather than
// forming one, so the SentencePiece dummy-prefix strip does NOT apply: that space
// belongs to the first token of the whole sequence, and these ids are not it.
//
// M-25: the serving loop decoded generated ids with Decode, which strips. On a
// dummy-prefix family (Llama-2/Mistral) a generation whose first token is `▁Paris`
// reached the client as "Paris" where OpenAI and llama.cpp both return " Paris".
// internal/gemmaapp decodes prompt+generation together for exactly this reason;
// this is the same correction without re-decoding the prompt every token.
//
// Byte-level tokenizers never strip, so this is identical to Decode for them.
func (t *Tokenizer) DecodeContinuation(ids []int) (string, error) {
	return t.decode(ids, false)
}

// DecodePiece decodes a single id to its display string — used for token
// streaming so the demo can print as it goes. A lone byte-fallback piece may
// be an incomplete UTF-8 sequence; callers that stream should buffer across
// calls (a demo concern, not the tokenizer's).
//
// It does NOT apply the whole-sequence dummy-prefix strip (audit M-13): that
// space belongs to the first token of the SEQUENCE, so stripping it per piece
// would drop the leading space of EVERY "▁word" token — a caller printing
// piece-by-piece would emit "Theanswerisfour". A streaming caller that wants the
// sequence's own leading space trimmed does it once on the assembled output.
func (t *Tokenizer) DecodePiece(id int) (string, error) {
	return t.decode([]int{id}, false)
}

// TokenText returns the raw surface bytes a single token id contributes when
// decoded — the per-token building block for byte-level constrained decoding
// (mapping the vocab onto a grammar). Unlike Decode it does NO whole-sequence
// post-processing: no SentencePiece leading-space strip, and no fusing of
// adjacent byte-fallback pieces. A byte-level piece is mapped through the byte
// decoder; a SentencePiece byte-fallback token yields its single raw byte; a
// normal SentencePiece piece has ▁ mapped to a space. Special tokens render as
// their literal surface form (so a grammar that forbids them masks them out).
// An out-of-range id returns nil.
func (t *Tokenizer) TokenText(id int) []byte {
	if id < 0 || id >= len(t.idToPiece) {
		return nil
	}
	if t.mode == modeByteLevel {
		var buf []byte
		for _, r := range t.idToPiece[id] {
			if b, ok := t.byteDecoder[r]; ok {
				buf = append(buf, b)
			} else {
				buf = utf8.AppendRune(buf, r) // defensive: non-byte-level rune
			}
		}
		return buf
	}
	// modeGemma: byte-fallback token → its raw byte; else ▁ → space.
	if b, ok := t.byteToVal[int32(id)]; ok {
		return []byte{b}
	}
	return []byte(strings.ReplaceAll(t.idToPiece[id], spaceMarker, " "))
}

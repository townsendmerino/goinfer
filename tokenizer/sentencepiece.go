package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
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
	vocab     map[string]int32 // piece → id
	idToPiece []string         // id → piece (len == vocab size)
	pairRank  map[bigram]int32 // BPE merge rank
	special   SpecialTokens

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
	return parseTokenizerJSON(raw, jsonPath)
}

// LoadJSONBytes parses a tokenizer from raw tokenizer.json bytes — the dir-less twin of
// Load. Used to load the tokenizer carried in a prequant .giw built from a SAFETENSORS
// model, whose tok half is the tokenizer.json itself (not a GGUF metadata blob, which is
// what LoadGGUFBytes expects). Self-contained tokenizer.json only (no sibling files).
func LoadJSONBytes(raw []byte) (*Tokenizer, error) {
	return parseTokenizerJSON(raw, "tokenizer.json")
}

// parseTokenizerJSON builds a Tokenizer from raw tokenizer.json bytes. jsonPath
// is the display path (for error messages) and locates any sibling files a
// byte-level pipeline references (via its directory). Split out from Load so the
// parse — the untrusted-input surface — is testable and fuzzable without a file
// on disk.
func parseTokenizerJSON(raw []byte, jsonPath string) (*Tokenizer, error) {
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
		if err := t.initByteLevel(&tj, filepath.Dir(jsonPath)); err != nil {
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
		{"<start_of_turn>", &t.special.StartOfTurn}, {"<end_of_turn>", &t.special.EndOfTurn},
	} {
		id, err := mustID(r.piece)
		if err != nil {
			return err
		}
		*r.dst = int(id)
	}
	return nil
}

// Encode turns text into token ids. If addBOS, prepend the BOS token (the
// generation prefill expects it for Gemma; byte-level families with no BOS
// ignore the flag). Added/special tokens written literally in the text are
// recognized and emitted as their own ids.
func (t *Tokenizer) Encode(text string, addBOS bool) ([]int, error) {
	// A vocab loaded without merges can DECODE but not encode — refuse rather than return a
	// silently unmerged (wrong) tokenization. See the merges note in gguf.go.
	if len(t.pairRank) == 0 {
		return nil, fmt.Errorf("tokenizer: this vocab was loaded without merge ranks (decode-only); cannot Encode")
	}
	if t.vocab == nil {
		return nil, fmt.Errorf("tokenizer.Encode: %w", errors.New("tokenizer not loaded"))
	}
	if t.mode == modeByteLevel {
		return t.encodeByteLevel(text, addBOS)
	}
	var out []int32
	if addBOS {
		out = append(out, int32(t.special.BOS))
	}

	gapStart := 0
	i := 0
	flushGap := func(end int) {
		if end > gapStart {
			out = append(out, t.bpe(t.normalizeGap(text[gapStart:end]))...)
		}
	}
	for i < len(text) {
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
func (t *Tokenizer) mergeSymbols(syms []string) []string {
	for len(syms) >= 2 {
		const maxRank = int32(1<<31 - 1)
		bestRank := maxRank
		bestI := -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.pairRank[bigram{syms[i], syms[i+1]}]; ok && r < bestRank {
				bestRank = r
				bestI = i
			}
		}
		if bestI < 0 {
			break
		}
		syms[bestI] += syms[bestI+1]
		syms = append(syms[:bestI+1], syms[bestI+2:]...)
	}
	return syms
}

// Decode turns token ids back into text: render each piece (with ▁ → space),
// fusing runs of byte-fallback pieces back into their raw UTF-8 bytes. Special
// tokens render as their literal surface form (e.g. "<eos>") — the generation
// loop is responsible for stopping at EOS, not Decode.
func (t *Tokenizer) Decode(ids []int) (string, error) {
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
	// SentencePiece decode strips the single leading space introduced by the
	// dummy prefix (Llama-2/Mistral); Gemma leaves it. This applies to the
	// rendered string as a whole, so callers streaming via DecodePiece should
	// decode the cumulative id slice (as the demo does), not piece-by-piece.
	if t.stripLeadingSpace {
		out = strings.TrimPrefix(out, " ")
	}
	return out, nil
}

// DecodePiece decodes a single id to its display string — used for token
// streaming so the demo can print as it goes. A lone byte-fallback piece may
// be an incomplete UTF-8 sequence; callers that stream should buffer across
// calls (a demo concern, not the tokenizer's).
func (t *Tokenizer) DecodePiece(id int) (string, error) {
	return t.Decode([]int{id})
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

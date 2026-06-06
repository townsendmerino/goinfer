package tokenizer

import (
	"fmt"
	"strings"

	"github.com/townsendmerino/aikit/embed"
	"golang.org/x/text/unicode/norm"
)

// GGUF tokenizer (G7 follow-up). A .gguf checkpoint carries its tokenizer in
// metadata — the vocab (tokenizer.ggml.tokens, id == array index), the BPE
// merges (tokenizer.ggml.merges, space-joined), per-token types, and the
// special-token ids — so LoadGGUF builds a Tokenizer with no sidecar
// tokenizer.json, letting a bare .gguf tokenize and chat end-to-end.
//
// Scope: the SentencePiece-style byte-fallback family (tokenizer.ggml.model ==
// "llama" — Llama-2/Mistral/TinyLlama and friends), which maps onto the same
// modeGemma merge-rank core with the ▁ dummy prefix. The merges live in
// metadata, so tokenization is merge-rank (not score-based) and reuses the
// shared BPE loop verbatim. The byte-level family ("gpt2" — Llama-3/Qwen/GPT-2)
// is a follow-up: same machinery as modeByteLevel, but its pretokenizer knobs
// (digit-run cap, NFC) come from tokenizer.ggml.pre and want a committed
// byte-level GGUF to parity-gate, which testdata doesn't have yet.
//
// Metadata reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

// GGUF tokenizer metadata keys (the llama.cpp convention).
const (
	ggufTokModel  = "tokenizer.ggml.model"
	ggufTokTokens = "tokenizer.ggml.tokens"
	ggufTokMerges = "tokenizer.ggml.merges"
	ggufTokTypes  = "tokenizer.ggml.token_type"
	ggufTokPre    = "tokenizer.ggml.pre"
	ggufTokBOS    = "tokenizer.ggml.bos_token_id"
	ggufTokEOS    = "tokenizer.ggml.eos_token_id"
	ggufTokUnk    = "tokenizer.ggml.unknown_token_id"
	ggufTokPad    = "tokenizer.ggml.padding_token_id"
	ggufTokPrefix = "tokenizer.ggml.add_space_prefix"
)

// ggml token types (tokenizer.ggml.token_type values). NORMAL and BYTE pieces
// go through BPE; the rest are added/special tokens matched literally in text.
const (
	ggufTokNormal      = 1
	ggufTokUnknown     = 2
	ggufTokControl     = 3
	ggufTokUserDefined = 4
	ggufTokUnused      = 5
	ggufTokByte        = 6
)

// LoadGGUF builds a Tokenizer from a .gguf file's embedded tokenizer metadata.
// It is the bare-checkpoint sibling of Load (which reads tokenizer.json): a
// .gguf is self-describing, so no other files are needed. It covers both GGUF
// tokenizer families: "llama" (SentencePiece byte-fallback — Llama-2/Mistral)
// and "gpt2" (byte-level — Llama-3/Qwen/GPT-2), dispatched on
// tokenizer.ggml.model.
func LoadGGUF(path string) (*Tokenizer, error) {
	// mmap, not heap-read: the tokenizer only needs the metadata at the head of
	// the file, so mapping avoids paging in the (multi-GB) weights entirely.
	g, err := embed.OpenGGUFMmap(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer.LoadGGUF: %w", err)
	}
	defer g.Close()
	t, err := fromGGUF(g)
	if err != nil {
		return nil, fmt.Errorf("tokenizer.LoadGGUF %s: %w", path, err)
	}
	return t, nil
}

// LoadGGUFBytes loads the tokenizer from an in-memory GGUF slice (e.g. an
// inflated //go:embed asset) — the bytes counterpart of LoadGGUF, touching no
// filesystem. raw only needs to live until this returns.
func LoadGGUFBytes(raw []byte) (*Tokenizer, error) {
	g, err := embed.OpenGGUFBytes(raw)
	if err != nil {
		return nil, fmt.Errorf("tokenizer.LoadGGUFBytes: %w", err)
	}
	defer g.Close()
	t, err := fromGGUF(g)
	if err != nil {
		return nil, fmt.Errorf("tokenizer.LoadGGUFBytes: %w", err)
	}
	return t, nil
}

func fromGGUF(g *embed.GGUFFile) (*Tokenizer, error) {
	model, ok := g.Str(ggufTokModel)
	if !ok {
		return nil, fmt.Errorf("missing %s (not a GGUF with an embedded tokenizer)", ggufTokModel)
	}
	if model == "gemma4" {
		model = "llama" // Gemma 4 uses the same SentencePiece byte-fallback tokenizer as the llama/gemma family
	}
	if model != "llama" && model != "gpt2" {
		return nil, fmt.Errorf("unsupported tokenizer model %q (have: llama [SPM byte-fallback], gpt2 [byte-level])", model)
	}

	tokens, err := ggufStringArray(g, ggufTokTokens)
	if err != nil {
		return nil, err
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("%s is empty", ggufTokTokens)
	}

	t := &Tokenizer{
		vocab:     make(map[string]int32, len(tokens)),
		idToPiece: tokens, // id == array index, contiguous and complete
		pairRank:  make(map[bigram]int32),
		byteToVal: make(map[int32]byte, 256),
	}
	for i, tok := range tokens {
		if _, exists := t.vocab[tok]; !exists { // first id wins on the rare dup
			t.vocab[tok] = int32(i)
		}
	}

	// Merge ranks: the merges array is the space-joined form ("▁ t" for SPM,
	// "Ġ Ġ" for byte-level); a piece never contains a literal space (it is the
	// ▁/Ġ marker), so the first space splits the pair unambiguously. Position in
	// the list is the priority.
	merges, err := ggufStringArray(g, ggufTokMerges)
	if err != nil {
		return nil, err
	}
	for i, m := range merges {
		l, r, ok := strings.Cut(m, " ")
		if !ok {
			return nil, fmt.Errorf("%s[%d] %q has no space separator", ggufTokMerges, i, m)
		}
		t.pairRank[bigram{l, r}] = int32(i)
	}

	// Special-token ids come straight from metadata (the surface forms vary by
	// model: <s>/</s>/<unk> for Llama-2). Chat-turn markers, if any, are
	// recognized by surface (ChatStyle() reads the same vocab entries).
	t.special = SpecialTokens{
		BOS:         ggufTokenID(g, ggufTokBOS),
		EOS:         ggufTokenID(g, ggufTokEOS),
		Pad:         ggufTokenID(g, ggufTokPad),
		StartOfTurn: -1,
		EndOfTurn:   -1,
	}
	for piece, dst := range map[string]*int{
		"<start_of_turn>": &t.special.StartOfTurn, "<end_of_turn>": &t.special.EndOfTurn,
		"<|im_start|>": &t.special.StartOfTurn, "<|im_end|>": &t.special.EndOfTurn,
	} {
		if id, ok := t.vocab[piece]; ok {
			*dst = int(id)
		}
	}

	// Per-family pipeline setup: SPM byte-fallback (llama) vs byte-level (gpt2).
	switch model {
	case "llama":
		setupGGUFSPM(t, g, tokens)
	case "gpt2":
		setupGGUFByteLevel(t, g)
	}

	// Added-vocabulary trie: every non-NORMAL, non-BYTE token (UNKNOWN /
	// CONTROL / USER_DEFINED) is split out of raw text before normalization —
	// HF's AddedVocabulary behavior. Falls back to the resolved specials when
	// the type array is absent.
	t.added = newAddedTrie()
	types := ggufIntArray(g, ggufTokTypes)
	if len(types) == len(tokens) {
		for i, ty := range types {
			switch ty {
			case ggufTokUnknown, ggufTokControl, ggufTokUserDefined:
				t.added.add(tokens[i], int32(i))
			}
		}
	} else {
		for _, id := range []int{t.special.BOS, t.special.EOS, t.special.Pad} {
			if id >= 0 && id < len(tokens) {
				t.added.add(tokens[id], int32(id))
			}
		}
	}

	return t, nil
}

// setupGGUFSPM configures the SentencePiece byte-fallback pipeline (modeGemma)
// for tokenizer.ggml.model == "llama": the <0xNN> byte map, the unknown token,
// and the ▁ dummy prefix (prepend on encode, strip one leading space on decode).
func setupGGUFSPM(t *Tokenizer, g *embed.GGUFFile, tokens []string) {
	t.mode = modeGemma

	// Byte-fallback "<0xNN>" pieces (token type BYTE) → the byte map.
	hasByte := false
	for b := range 256 {
		p := fmt.Sprintf("<0x%02X>", b)
		t.bytePiece[b] = p
		if id, ok := t.vocab[p]; ok {
			t.byteToVal[id] = byte(b)
			hasByte = true
		}
	}
	t.byteFallback = hasByte

	// Unknown token: id from metadata (its surface is whatever tokens[unk] is),
	// falling back to the conventional "<unk>".
	t.unkPiece, t.unkID = "<unk>", 0
	if unk := ggufTokenID(g, ggufTokUnk); unk >= 0 && unk < len(tokens) {
		t.unkID, t.unkPiece = int32(unk), tokens[unk]
	} else if id, ok := t.vocab["<unk>"]; ok {
		t.unkID = id
	}

	// SentencePiece dummy prefix: "llama" SPM prepends a ▁ (and strips one
	// leading space on decode) unless add_space_prefix says otherwise.
	t.prependSpace = true
	if v, ok := g.Metadata[ggufTokPrefix].(bool); ok {
		t.prependSpace = v
	}
	t.stripLeadingSpace = t.prependSpace
}

// setupGGUFByteLevel configures the GPT-2/Llama-3/Qwen byte-level pipeline
// (modeByteLevel) for tokenizer.ggml.model == "gpt2": the byte↔rune tables plus
// the pretokenizer knobs (digit-run cap, NFC, ignore_merges) selected from
// tokenizer.ggml.pre — the GGUF analogue of reading them from tokenizer.json's
// normalizer/pre_tokenizer.
func setupGGUFByteLevel(t *Tokenizer, g *embed.GGUFFile) {
	t.mode = modeByteLevel
	t.byteEncoder, t.byteDecoder = buildByteLevelTables()
	pre, _ := g.Str(ggufTokPre)
	t.maxDigits, t.normForm, t.normOn, t.ignoreMerges, t.splitDigits = byteLevelKnobs(pre)
}

// byteLevelKnobs maps a tokenizer.ggml.pre identifier to the byte-level pipeline
// knobs (digit-run cap, normalization form + on, ignore_merges, individual-digit
// pre-split). The values reproduce what Load derives from each family's
// tokenizer.json: Llama-3 groups digits in runs of ≤3 with no normalizer and
// ignore_merges; Qwen takes one digit, NFC-normalizes, and honors merges; Mellum2
// has no normalizer and a Digits{individual_digits} pretokenizer (each digit
// isolated, so a leading space never attaches to one); GPT-2 takes one digit, no
// NFC, and honors merges. Unknown pre falls back to the GPT-2-like defaults.
func byteLevelKnobs(pre string) (maxDigits int, form norm.Form, normOn, ignoreMerges, splitDigits bool) {
	switch pre {
	case "llama-bpe", "llama3", "llama-v3":
		return 3, norm.NFC, false, true, false
	case "qwen2", "qwen2.5", "qwen":
		return 1, norm.NFC, true, false, false
	case "mellum2":
		return 1, norm.NFC, false, false, true
	default: // "gpt-2", "default", "", and unrecognized
		return 1, norm.NFC, false, false, false
	}
}

// ggufStringArray reads a GGUF metadata array of strings.
func ggufStringArray(g *embed.GGUFFile, key string) ([]string, error) {
	arr, ok := g.Metadata[key].([]any)
	if !ok {
		return nil, fmt.Errorf("%s missing or not an array", key)
	}
	out := make([]string, len(arr))
	for i, v := range arr {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("%s[%d] is %T, want string", key, i, v)
		}
		out[i] = s
	}
	return out, nil
}

// ggufIntArray reads a GGUF metadata array of integers (any width), returning
// nil if the key is absent or not an array — callers treat that as "no data".
func ggufIntArray(g *embed.GGUFFile, key string) []int {
	arr, ok := g.Metadata[key].([]any)
	if !ok {
		return nil
	}
	out := make([]int, len(arr))
	for i, v := range arr {
		switch n := v.(type) {
		case int8:
			out[i] = int(n)
		case int16:
			out[i] = int(n)
		case int32:
			out[i] = int(n)
		case int64:
			out[i] = int(n)
		case uint8:
			out[i] = int(n)
		case uint16:
			out[i] = int(n)
		case uint32:
			out[i] = int(n)
		case uint64:
			out[i] = int(n)
		}
	}
	return out
}

// ggufTokenID reads an integer special-token-id metadata value, or -1 ("none").
func ggufTokenID(g *embed.GGUFFile, key string) int {
	if v, ok := g.Uint(key); ok {
		return int(v)
	}
	return -1
}

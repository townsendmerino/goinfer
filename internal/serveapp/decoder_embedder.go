package serveapp

// Decoder-as-embedder (docs/completed/task-decoder-as-embedder.md).
//
// qwen3-embedding and embeddinggemma are causal decoders used as EMBEDDERS. goinfer already runs
// those decoders and already serves /v1/embeddings against an aikit encoder; the only genuinely new
// piece is a pooling head + instruction-prefix convention:
//
//	decoder forward → last-token pool (decoder.HiddenLast) → out the existing /v1/embeddings
//
// This type satisfies aikit's encoder.Encoder structurally (Encode / EncodeBatch / HiddenDim), so
// it drops straight into server.embed and every downstream behavior — input_type asymmetry,
// dimensions truncate+renormalize, encoding_format, L2 normalization, usage counting — is unchanged.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// Qwen3-Embedding conventions. Read from the model's own config files AND — the part config-reading
// alone gets wrong — from what sentence-transformers actually feeds the model:
//   - config_sentence_transformers.json prompts: query carries the Instruct preamble, document "".
//   - 1_Pooling/config.json: pooling_mode_lasttoken=true, include_prompt=true — so the query prompt
//     is part of the pooled input and is NOT stripped before pooling.
//   - tokenizer_config.json: add_bos_token=false, add_eos_token ABSENT -> no BOS, and the plain
//     tokenizer call appends nothing.
//   - BUT sentence-transformers appends <|endoftext|> (151643) to EVERY input. Nothing in the
//     configs or the model card says so — `model.tokenize()` had to be inspected to see it. This is
//     not cosmetic: last-token pooling pools THAT token, so omitting it pools the final content
//     token instead and yields a plausible, semantically-ordered, but WRONG vector (cosine ~0.4-0.8
//     vs the reference, while retrieval still looks fine). Note it is <|endoftext|>, NOT the
//     configured eos_token <|im_end|> (151645) — reading eos_token would also have been wrong.
const (
	qwen3EmbedQueryPrompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
	qwen3EmbedDocPrompt   = ""
	// qwen3EmbedEOD is appended to every sequence and is the position last-token pooling reads.
	qwen3EmbedEOD = "<|endoftext|>"
)

// decoderEmbedder adapts a loaded goinfer decoder into the embedder seam.
type decoderEmbedder struct {
	// mu serializes EVERYTHING. The /v1/embeddings handler deliberately runs without the server
	// mutex because an aikit encoder is goroutine-safe for concurrent Encode — a DECODER is not:
	// it has one shared KV cache and one decode scratch. Without this, parallel embedding requests
	// would interleave writes into the same cache and return plausible-but-wrong vectors (and race).
	mu sync.Mutex

	m   *decoder.Model
	tk  *tokenizer.Tokenizer
	dim int

	queryPrompt string // prepended when isQuery (pooled with the text — include_prompt: true)
	docPrompt   string // prepended otherwise ("" for Qwen3-Embedding)
	appendID    int    // token appended to every sequence (<0 = none); THIS is what gets pooled
	maxTokens   int    // truncate to this many tokens (0 = no limit), mirroring HF truncation=True
}

// newDecoderEmbedder builds the embedder and resolves the appended EOD token id.
//
// Construct through this, never as a struct literal: appendID's ZERO value is 0, which is a real
// token id, so a literal silently appends token 0 and pools it. (That is exactly what a first cut
// of the tests did — the tokenizer-agreement gate caught it.)
func newDecoderEmbedder(m *decoder.Model, tk *tokenizer.Tokenizer, queryPrompt, docPrompt string) *decoderEmbedder {
	appendID := -1
	if id, ok := tk.TokenID(qwen3EmbedEOD); ok {
		appendID = id
	}
	return &decoderEmbedder{
		m:           m,
		tk:          tk,
		dim:         m.Config().HiddenDim,
		queryPrompt: queryPrompt,
		docPrompt:   docPrompt,
		appendID:    appendID,
		// BOUNDED BY THE MODEL'S CONTEXT, always. The reference imposes nothing shorter than the
		// tokenizer's model_max_length (Qwen3-Embedding ships no sentence_bert_config.json), but
		// "nothing shorter" is not "unbounded": HiddenLast preallocates KV for len(ids) positions
		// and runs a sequential per-token forward with no context, so 0 here meant the ONLY bound
		// was C-21's 1 MiB byte cap — about 500k tokens of short words. For a 28-layer, kvDim-1024
		// embedder that is ~114 GB of KV and attention over up to 500k keys per token: hours of one
		// pinned core, every other /v1/embeddings request blocked behind the embed mutex, and then
		// an OOM kill. Uncancellable, and un-queued.
		//
		// Even a legitimate 200 KB document (~45k tokens) silently exceeded Qwen3-Embedding's 40k
		// window and returned a vector pooled from out-of-range RoPE positions — plausible, and
		// wrong. MaxPositions is HF's truncation=True semantics, which is what the aikit encoder
		// path already does; only this one was unbounded (audit-2026-09-02 C-07, an incomplete
		// closure of the 2026-08-05 audit's C-21).
		maxTokens: m.Config().MaxPositions,
	}
}

// loadDecoderEmbedder wires a causal decoder (.gguf) in as the embedder. Selected by -embed-model
// pointing at a FILE rather than an HF directory (see loadEncoder).
//
// embedTok is deliberately left nil: this embedder counts tokens with its own decoder tokenizer
// through embedTokenCounter, since an aikit embed.Tokenizer cannot tokenize for a decoder.
func (s *server) loadDecoderEmbedder(cfg config) error {
	t0 := time.Now()
	tk, err := loadDecoderTokenizer(cfg.embedPath)
	if err != nil {
		return fmt.Errorf("load embedding tokenizer (%s): %w", cfg.embedPath, err)
	}
	m, err := decoder.Load(cfg.embedPath, decoder.Options{})
	if err != nil {
		return fmt.Errorf("load embedding model (%s): %w", cfg.embedPath, err)
	}
	name := cfg.embedName
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(cfg.embedPath), ".gguf")
	}
	e := newDecoderEmbedder(m, tk, qwen3EmbedQueryPrompt, qwen3EmbedDocPrompt)
	if e.appendID < 0 {
		fmt.Fprintf(os.Stderr, "warning: embedding model %q has no %s token — last-token pooling will pool the final content token, which will NOT match the reference\n",
			cfg.embedPath, qwen3EmbedEOD)
	}
	s.embed, s.embedTok, s.embedID, s.embedDim = e, nil, name, e.HiddenDim()
	// Not truncatable, explicitly. Qwen3-Embedding documents MRL support, but nothing here has
	// MEASURED it the way aikit's coverage gate measures its own rows, and an unmeasured floor is
	// exactly the guess resolveDimensions exists to refuse. Refusing `dimensions` is the safe
	// direction; certifying a floor (aikit's paraphrase-pair-recall method) is what would change it.
	s.embedMRLMin = 0
	fmt.Fprintf(os.Stderr, "loaded decoder-backed embedding model %q (dim %d, last-token pooling) in %s\n",
		name, s.embedDim, time.Since(t0).Round(time.Millisecond))
	return nil
}

// HiddenDim is the embedding width — the decoder's hidden size, since we pool the residual stream.
func (e *decoderEmbedder) HiddenDim() int { return e.dim }

// Encode embeds one text. isQuery selects the instruction prefix (the query/document asymmetry the
// handler already maps input_type onto). The vector is raw; the handler L2-normalizes it.
func (e *decoderEmbedder) Encode(text string, isQuery bool) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.encodeLocked(text, isQuery)
}

// EncodeBatch embeds each text in turn.
//
// concurrency is deliberately IGNORED (i.e. clamped to 1): the aikit encoder can fan out because it
// is stateless per call, but this embedder is one decoder with one KV cache. Running texts in
// parallel would corrupt that shared state, so the batch is sequential — the honest contract, rather
// than accepting a concurrency argument we cannot satisfy.
func (e *decoderEmbedder) EncodeBatch(texts []string, isQueries []bool, concurrency int) ([][]float32, error) {
	if len(isQueries) != len(texts) {
		return nil, fmt.Errorf("decoder embedder: %d texts but %d isQuery flags", len(texts), len(isQueries))
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v, err := e.encodeLocked(t, isQueries[i])
		if err != nil {
			return nil, fmt.Errorf("embed input %d: %w", i, err)
		}
		out[i] = v
	}
	return out, nil
}

// CountTokens reports how many tokens this embedder actually feeds the model for text — prefix
// included, truncation applied. See embedTokenCounter in embeddings.go for why this exists.
func (e *decoderEmbedder) CountTokens(text string, isQuery bool) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	ids, err := e.tokenize(text, isQuery)
	if err != nil {
		return 0
	}
	return len(ids)
}

// encodeLocked is Encode's body; callers hold e.mu.
func (e *decoderEmbedder) encodeLocked(text string, isQuery bool) ([]float32, error) {
	ids, err := e.tokenize(text, isQuery)
	if err != nil {
		return nil, err
	}
	return e.m.HiddenLast(ids)
}

// tokenize applies the instruction prefix and encodes WITHOUT special tokens (Qwen3-Embedding adds
// neither BOS nor EOS), truncating from the end exactly as HF's truncation=True does. Truncation
// moves the pooled position, which is correct: last-token pooling pools whatever the final token is.
func (e *decoderEmbedder) tokenize(text string, isQuery bool) ([]int, error) {
	prompt := e.docPrompt
	if isQuery {
		prompt = e.queryPrompt
	}
	ids, err := e.tk.Encode(prompt+text, false) // addBOS=false
	if err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		// HiddenLast needs a position to pool. Say so rather than returning a zero vector that
		// would look like a legitimate embedding.
		return nil, fmt.Errorf("decoder embedder: input tokenized to zero tokens (empty input?)")
	}
	// Truncate BEFORE appending, reserving the slot, so the appended token is never the thing
	// truncation drops — it must stay last, because it is the pooled position.
	if e.maxTokens > 0 {
		room := e.maxTokens
		if e.appendID >= 0 {
			room--
		}
		if room > 0 && len(ids) > room {
			ids = ids[:room]
		}
	}
	if e.appendID >= 0 {
		ids = append(ids, e.appendID)
	}
	return ids, nil
}

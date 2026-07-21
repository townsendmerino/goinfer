package main

// Decoder-as-embedder (docs/task-decoder-as-embedder.md).
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

// Qwen3-Embedding conventions, verified against the model's own config files (not by reputation —
// see the task doc's warning about that):
//   - config_sentence_transformers.json prompts: query carries the Instruct preamble, document "".
//   - 1_Pooling/config.json: pooling_mode_lasttoken=true, include_prompt=true — so the query prompt
//     is part of the pooled input and is NOT stripped before pooling.
//   - tokenizer_config.json: add_bos_token=false and add_eos_token absent; the model card appends no
//     EOS/EOD in either the sentence-transformers or raw-transformers example. So the sequence fed to
//     the model is exactly the (prompt+text) tokens — no special tokens on either end.
const (
	qwen3EmbedQueryPrompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
	qwen3EmbedDocPrompt   = ""
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
	maxTokens   int    // truncate to this many tokens (0 = no limit), mirroring HF truncation=True
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
	e := &decoderEmbedder{
		m:           m,
		tk:          tk,
		dim:         m.Config().HiddenDim,
		queryPrompt: qwen3EmbedQueryPrompt,
		docPrompt:   qwen3EmbedDocPrompt,
		// No truncation by default: Qwen3-Embedding ships no sentence_bert_config.json, so the
		// reference imposes nothing shorter than the tokenizer's model_max_length (131072).
		maxTokens: 0,
	}
	s.embed, s.embedTok, s.embedID, s.embedDim = e, nil, name, e.HiddenDim()
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
	if e.maxTokens > 0 && len(ids) > e.maxTokens {
		ids = ids[:e.maxTokens]
	}
	return ids, nil
}

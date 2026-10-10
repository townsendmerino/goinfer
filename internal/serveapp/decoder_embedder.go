package serveapp

// Decoder-as-embedder (docs/completed/task-decoder-as-embedder.md).
//
// qwen3-embedding and embeddinggemma are causal decoders used as embedders. Beyond running the decoder and serving
// /v1/embeddings, the only new piece is a pooling head and an instruction-prefix convention: decoder forward,
// last-token pool (decoder.HiddenLast), out through the existing /v1/embeddings. decoderEmbedder satisfies aikit's
// encoder.Encoder structurally (Encode / EncodeBatch / HiddenDim), so it drops into server.embed and every downstream
// behavior (input_type asymmetry, dimensions truncate+renormalize, encoding_format, L2 normalization, usage counting)
// is unchanged.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/modelload"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// Qwen3-Embedding conventions, read from the model's config files and from what sentence-transformers actually feeds the
// model:
//   - config_sentence_transformers.json prompts: query carries the Instruct preamble, document "".
//   - 1_Pooling/config.json: pooling_mode_lasttoken=true, include_prompt=true, so the query prompt is part of the pooled
//     input and is not stripped before pooling.
//   - tokenizer_config.json: add_bos_token=false, add_eos_token absent: no BOS, and the plain tokenizer call appends
//     nothing.
//   - sentence-transformers appends <|endoftext|> (151643) to every input, which no config or model card says. It is not
//     cosmetic: last-token pooling pools that token, so omitting it pools the final content token and yields a plausible,
//     semantically ordered, but wrong vector. It is <|endoftext|>, not the configured eos_token <|im_end|> (151645).
const (
	qwen3EmbedQueryPrompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
	qwen3EmbedDocPrompt   = ""
	// qwen3EmbedEOD is appended to every sequence and is the position last-token pooling reads.
	qwen3EmbedEOD = "<|endoftext|>"
)

// decoderEmbedder adapts a loaded goinfer decoder into the embedder seam.
type decoderEmbedder struct {
	// mu serializes everything. The /v1/embeddings handler runs without the server mutex because an aikit encoder is
	// goroutine-safe for concurrent Encode, but a decoder is not: it has one shared KV cache and one decode scratch,
	// so parallel embedding requests would interleave writes into the same cache and return plausible but wrong
	// vectors.
	mu sync.Mutex

	m   *decoder.Model
	tk  *tokenizer.Tokenizer
	dim int

	queryPrompt string // prepended when isQuery (pooled with the text — include_prompt: true)
	docPrompt   string // prepended otherwise ("" for Qwen3-Embedding)
	appendID    int    // token appended to every sequence (<0 = none); THIS is what gets pooled
	maxTokens   int    // truncate to this many tokens (0 = no limit), mirroring HF truncation=True
}

// newDecoderEmbedder builds the embedder and resolves the appended EOD token id. Construct through this, never as a
// struct literal: appendID's zero value is 0, a real token id, so a literal silently appends token 0 and pools it.
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
		// Bounded by the model's context, always. The reference imposes nothing shorter than the tokenizer's
		// model_max_length, but HiddenLast preallocates KV for len(ids) positions and runs a sequential per-token
		// forward with no context, so an unbounded value leaves only the request byte cap (about 500k tokens of short
		// words): hours of one pinned core, every other /v1/embeddings request blocked behind the embed mutex, then
		// an OOM kill, uncancellable and un-queued. A long legitimate document would also exceed the model's window
		// and return a vector pooled from out-of-range RoPE positions: plausible, and wrong. MaxPositions is HF's
		// truncation=True semantics, as the aikit encoder path already does.
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
	tk, err := modelload.Tokenizer(cfg.embedPath)
	if err != nil {
		return fmt.Errorf("load embedding tokenizer (%s): %w", cfg.embedPath, err)
	}
	// Backend and Quant must be set here: left zero-valued, the embedder always loads CPU-only whatever -backend
	// says, and the resident HiddenLast path (decoder/embed.go) is unreachable from serve. decoder.Load builds the
	// resident runner from Options.Backend alone, so this is sufficient, with no separate resident-build step, as for
	// chat models (modelSpec.options). A resident embedder gets its own resBusy CAS (decoder/model.go), like a second
	// resident chat model.
	//
	// cfg.embedQuant is not passed through as decoder.Options.Quant: the vocabularies differ (-embed-quant's
	// "f32"|"q8" against Options.Quant's ""|"int8"|"int8int8"|"int4"), and parseQuant hard-errors on a string it does
	// not recognize, so passing "q8" straight through would make -embed-quant q8 fail to load. "q8" maps to Quant's
	// "int8" (weight-only per-row), the precision class the aikit encoder's LoadQ8 path uses.
	quant := ""
	if strings.EqualFold(cfg.embedQuant, "q8") {
		quant = "int8"
	}
	m, err := decoder.Load(cfg.embedPath, decoder.Options{Backend: cfg.load.Backend, BackendAuto: cfg.load.Auto != nil, Quant: quant,
		ExactPrefill: cfg.load.ExactPrefill, Knobs: cfg.load.Knobs()}) // the same prompt-ingestion flags as chat models
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
	// Not truncatable, explicitly. Qwen3-Embedding documents MRL support, but nothing here has measured it the way
	// aikit's coverage gate measures its own rows, and an unmeasured floor is the guess resolveDimensions exists to
	// refuse. Certifying a floor (aikit's paraphrase-pair-recall method) is what would change it.
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
	v, _, err := e.encodeLocked(text, isQuery)
	return v, err
}

// EncodeBatch embeds each text in turn. concurrency is ignored (clamped to 1): the aikit encoder can fan out because
// it is stateless per call, but this embedder is one decoder with one KV cache, so running texts in parallel would
// corrupt that shared state.
func (e *decoderEmbedder) EncodeBatch(texts []string, isQueries []bool, concurrency int) ([][]float32, error) {
	if len(isQueries) != len(texts) {
		return nil, fmt.Errorf("decoder embedder: %d texts but %d isQuery flags", len(texts), len(isQueries))
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v, _, err := e.encodeLocked(t, isQueries[i])
		if err != nil {
			return nil, fmt.Errorf("embed input %d: %w", i, err)
		}
		out[i] = v
	}
	return out, nil
}

// EncodeBatchCounted is EncodeBatch plus each input's token count, read off the tokenize call encodeLocked already
// makes instead of a second pass over the text. embedBatchCounter (embeddings.go) is the optional capability the
// handler prefers; encoders that do not implement it (the aikit-embed.Tokenizer path) are unaffected.
func (e *decoderEmbedder) EncodeBatchCounted(texts []string, isQueries []bool, concurrency int) ([][]float32, []int, error) {
	if len(isQueries) != len(texts) {
		return nil, nil, fmt.Errorf("decoder embedder: %d texts but %d isQuery flags", len(texts), len(isQueries))
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	vecs := make([][]float32, len(texts))
	counts := make([]int, len(texts))
	for i, t := range texts {
		v, ids, err := e.encodeLocked(t, isQueries[i])
		if err != nil {
			return nil, nil, fmt.Errorf("embed input %d: %w", i, err)
		}
		vecs[i] = v
		counts[i] = len(ids)
	}
	return vecs, counts, nil
}

// CountTokens reports how many tokens this embedder feeds the model for text, prefix included and truncation applied
// (see embedTokenCounter in embeddings.go). For callers that only want a count; EncodeBatchCounted is the byproduct
// path for the /v1/embeddings handler.
func (e *decoderEmbedder) CountTokens(text string, isQuery bool) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	ids, err := e.tokenize(text, isQuery)
	if err != nil {
		return 0
	}
	return len(ids)
}

// encodeLocked is Encode's body; callers hold e.mu. It returns the token ids with the vector so EncodeBatchCounted
// can report their count without a second tokenize pass.
func (e *decoderEmbedder) encodeLocked(text string, isQuery bool) ([]float32, []int, error) {
	ids, err := e.tokenize(text, isQuery)
	if err != nil {
		return nil, nil, err
	}
	v, err := e.m.HiddenLast(ids)
	if err != nil {
		return nil, nil, err
	}
	return v, ids, nil
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
	ids = truncateForContext(ids, e.maxTokens, e.appendID)
	if e.appendID >= 0 {
		ids = append(ids, e.appendID)
	}
	return ids, nil
}

// truncateForContext is tokenize's truncation arithmetic, pulled out so a test calls the same code the request path
// runs instead of re-deriving it beside it. It truncates before the caller appends appendID, reserving its slot, so
// the appended token is never what truncation drops: it must stay last, because it is the pooled position.
func truncateForContext(ids []int, maxTokens, appendID int) []int {
	if maxTokens <= 0 {
		return ids
	}
	room := maxTokens
	if appendID >= 0 {
		room--
	}
	if room > 0 && len(ids) > room {
		ids = ids[:room]
	}
	return ids
}

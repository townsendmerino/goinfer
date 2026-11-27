package embedders

import (
	"context"
	"fmt"
	"sync"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// Qwen3-Embedding instruction-prefix and pooled-EOD conventions.
const (
	Qwen3EmbedQueryPrompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
	Qwen3EmbedDocPrompt   = ""
	Qwen3EmbedEOD         = "<|endoftext|>"
)

// DecoderEmbedder adapts a loaded goinfer causal decoder into a serialized,
// context-cancellable last-token-pooling embedder.
type DecoderEmbedder struct {
	slot     chan struct{}
	slotOnce sync.Once

	m   *decoder.Model
	tk  *tokenizer.Tokenizer
	dim int

	queryPrompt string
	docPrompt   string
	appendID    int
	maxTokens   int
}

// NewDecoderEmbedder constructs a DecoderEmbedder bounded by m.Config().MaxPositions
// and resolves any Qwen3EmbedEOD token in tk.
func NewDecoderEmbedder(m *decoder.Model, tk *tokenizer.Tokenizer, queryPrompt, docPrompt string) *DecoderEmbedder {
	appendID := -1
	if id, ok := tk.TokenID(Qwen3EmbedEOD); ok {
		appendID = id
	}
	return &DecoderEmbedder{
		m:           m,
		tk:          tk,
		dim:         m.Config().HiddenDim,
		queryPrompt: queryPrompt,
		docPrompt:   docPrompt,
		appendID:    appendID,
		maxTokens:   m.Config().MaxPositions,
	}
}

// HiddenDim returns the decoder's hidden size.
func (e *DecoderEmbedder) HiddenDim() int { return e.dim }

// Model returns the underlying decoder model.
func (e *DecoderEmbedder) Model() *decoder.Model { return e.m }

// Tokenizer returns the underlying decoder tokenizer.
func (e *DecoderEmbedder) Tokenizer() *tokenizer.Tokenizer { return e.tk }

// AppendID returns the EOD token id appended before pooling, or -1 if absent.
func (e *DecoderEmbedder) AppendID() int { return e.appendID }

// MaxTokens returns the context-window token ceiling applied before pooling.
func (e *DecoderEmbedder) MaxTokens() int { return e.maxTokens }

// Encode embeds a single text under the query or document prompt.
func (e *DecoderEmbedder) Encode(text string, isQuery bool) ([]float32, error) {
	ctx := context.Background()
	if err := e.Acquire(ctx); err != nil {
		return nil, err
	}
	defer e.Release()
	v, _, err := e.encodeLocked(ctx, text, isQuery)
	return v, err
}

// EncodeBatch embeds each text sequentially under the shared decoder slot.
func (e *DecoderEmbedder) EncodeBatch(texts []string, isQueries []bool, concurrency int) ([][]float32, error) {
	if len(isQueries) != len(texts) {
		return nil, fmt.Errorf("decoder embedder: %d texts but %d isQuery flags", len(texts), len(isQueries))
	}
	ctx := context.Background()
	if err := e.Acquire(ctx); err != nil {
		return nil, err
	}
	defer e.Release()
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v, _, err := e.encodeLocked(ctx, t, isQueries[i])
		if err != nil {
			return nil, fmt.Errorf("embed input %d: %w", i, err)
		}
		out[i] = v
	}
	return out, nil
}

// EncodeBatchCounted embeds each text and returns per-input token counts from the same pass.
func (e *DecoderEmbedder) EncodeBatchCounted(texts []string, isQueries []bool, concurrency int) ([][]float32, []int, error) {
	return e.EncodeBatchCountedContext(context.Background(), texts, isQueries, concurrency)
}

// EncodeBatchCountedContext embeds each text under ctx, returning ctx.Err() if cancelled while waiting or running.
func (e *DecoderEmbedder) EncodeBatchCountedContext(ctx context.Context, texts []string, isQueries []bool, concurrency int) ([][]float32, []int, error) {
	if len(isQueries) != len(texts) {
		return nil, nil, fmt.Errorf("decoder embedder: %d texts but %d isQuery flags", len(texts), len(isQueries))
	}
	if err := e.Acquire(ctx); err != nil {
		return nil, nil, err
	}
	defer e.Release()
	vecs := make([][]float32, len(texts))
	counts := make([]int, len(texts))
	for i, t := range texts {
		v, ids, err := e.encodeLocked(ctx, t, isQueries[i])
		if err != nil {
			return nil, nil, fmt.Errorf("embed input %d: %w", i, err)
		}
		vecs[i] = v
		counts[i] = len(ids)
	}
	return vecs, counts, nil
}

// CountTokens reports how many tokens e feeds the model for text, including prefix and truncation.
func (e *DecoderEmbedder) CountTokens(text string, isQuery bool) int {
	if e.Acquire(context.Background()) != nil {
		return 0
	}
	defer e.Release()
	ids, err := e.Tokenize(text, isQuery)
	if err != nil {
		return 0
	}
	return len(ids)
}

// Acquire takes the embedder's single execution slot or returns ctx.Err().
func (e *DecoderEmbedder) Acquire(ctx context.Context) error {
	e.slotOnce.Do(func() { e.slot = make(chan struct{}, 1) })
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case e.slot <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Release returns the slot taken by Acquire.
func (e *DecoderEmbedder) Release() { <-e.slot }

func (e *DecoderEmbedder) encodeLocked(ctx context.Context, text string, isQuery bool) ([]float32, []int, error) {
	if e.m == nil {
		return nil, nil, decoder.ErrClosed
	}
	ids, err := e.Tokenize(text, isQuery)
	if err != nil {
		return nil, nil, err
	}
	v, err := e.m.HiddenLastContext(ctx, ids)
	if err != nil {
		return nil, nil, err
	}
	return v, ids, nil
}

// Tokenize applies the query/document prefix, encodes without BOS/EOS, truncates
// to MaxTokens, and appends AppendID when non-negative.
func (e *DecoderEmbedder) Tokenize(text string, isQuery bool) ([]int, error) {
	prompt := e.docPrompt
	if isQuery {
		prompt = e.queryPrompt
	}
	ids, err := e.tk.Encode(prompt+text, false)
	if err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("decoder embedder: input tokenized to zero tokens (empty input?)")
	}
	ids = TruncateForContext(ids, e.maxTokens, e.appendID)
	if e.appendID >= 0 {
		ids = append(ids, e.appendID)
	}
	return ids, nil
}

// Close closes the underlying decoder model under the execution slot.
func (e *DecoderEmbedder) Close() error {
	_ = e.Acquire(context.Background())
	defer e.Release()
	if e.m == nil {
		return nil
	}
	err := e.m.Close()
	e.m = nil
	return err
}

// TruncateForContext truncates ids to leave room for appendID within maxTokens
// so the pooled final token is never dropped by truncation.
func TruncateForContext(ids []int, maxTokens, appendID int) []int {
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

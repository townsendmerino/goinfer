package embeddinggemma2

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// MaxTokens is the context the model card gives for every modality (8,192). An input longer than this is refused,
// not cut: a silently truncated input would embed a different text than the caller sent.
const MaxTokens = 8192

// MatryoshkaWidths are the embedding widths the model was trained to be truncated to (the model card's 768, 512, 256
// and 128). A server offers exactly these for OpenAI's `dimensions`.
var MatryoshkaWidths = []int{768, 512, 256, 128}

// Encoder is an EmbeddingGemma 2 model with its tokenizer and its named task prompts
// (config_sentence_transformers.json's `prompts`). It implements aikit's encoder.Encoder (Encode, EncodeBatch,
// HiddenDim), where isQuery selects the checkpoint's own "query" prompt and false its "document" prompt, and offers
// EncodeTasks for a caller that names the prompt, or none. It is safe for concurrent use.
type Encoder struct {
	m       *Model
	tok     *tokenizer.Tokenizer
	prompts map[string]string
	bos     int
	eos     int
}

// LoadEncoder loads the model, tokenizer.json and the prompt table from an HF checkpoint directory.
func LoadEncoder(dir string) (*Encoder, error) {
	m, err := Load(dir)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.Load(dir)
	if err != nil {
		return nil, fmt.Errorf("embeddinggemma2: tokenizer: %w", err)
	}
	bos, okB := tok.TokenID("<bos>")
	eos, okE := tok.TokenID("<eos>")
	if !okB || !okE {
		return nil, fmt.Errorf("embeddinggemma2: tokenizer has no <bos>/<eos>")
	}
	prompts := map[string]string{}
	if raw, err := os.ReadFile(filepath.Join(dir, "config_sentence_transformers.json")); err == nil {
		var st struct {
			Prompts map[string]string `json:"prompts"`
		}
		if err := json.Unmarshal(raw, &st); err != nil {
			return nil, fmt.Errorf("embeddinggemma2: config_sentence_transformers.json: %w", err)
		}
		prompts = st.Prompts
	}
	return &Encoder{m: m, tok: tok, prompts: prompts, bos: bos, eos: eos}, nil
}

// Model returns the underlying text encoder.
func (e *Encoder) Model() *Model { return e.m }

// HiddenDim is the embedding width (768).
func (e *Encoder) HiddenDim() int { return e.m.Dim() }

// PromptNames lists the checkpoint's named prompts, sorted.
func (e *Encoder) PromptNames() []string {
	names := make([]string, 0, len(e.prompts))
	for n := range e.prompts {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// PromptText returns the text a named prompt prepends. The empty name is no prompt.
func (e *Encoder) PromptText(name string) (string, error) {
	if name == "" {
		return "", nil
	}
	if p, ok := e.prompts[name]; ok {
		return p, nil
	}
	return "", fmt.Errorf("unknown task %q; this model's tasks are %s, or none", name, strings.Join(e.PromptNames(), ", "))
}

// Tokenize returns the ids the model sees for text under the named prompt: <bos>, the prompt and the text as one
// string, <eos>, as the checkpoint's tokenizer post-processor writes them. sentence-transformers mean-pools over all of
// them, the prompt and both special tokens included.
func (e *Encoder) Tokenize(text, prompt string) ([]int, error) {
	p, err := e.PromptText(prompt)
	if err != nil {
		return nil, err
	}
	ids, err := e.tok.Encode(p+text, false)
	if err != nil {
		return nil, err
	}
	out := make([]int, 0, len(ids)+2)
	out = append(out, e.bos)
	out = append(out, ids...)
	out = append(out, e.eos)
	if len(out) > MaxTokens {
		return nil, fmt.Errorf("input is %d tokens, over the model's %d-token context", len(out), MaxTokens)
	}
	return out, nil
}

// EmbedText is the sentence embedding of text under the named prompt ("" for none), unit length.
func (e *Encoder) EmbedText(text, prompt string) ([]float32, int, error) {
	ids, err := e.Tokenize(text, prompt)
	if err != nil {
		return nil, 0, err
	}
	v, err := e.m.Embed(ids)
	return v, len(ids), err
}

// EncodeTasks embeds every text under one named prompt ("" for none) and reports each input's token count.
func (e *Encoder) EncodeTasks(texts []string, prompt string) ([][]float32, []int, error) {
	vecs := make([][]float32, len(texts))
	counts := make([]int, len(texts))
	for i, t := range texts {
		v, n, err := e.EmbedText(t, prompt)
		if err != nil {
			return nil, nil, fmt.Errorf("input %d: %w", i, err)
		}
		vecs[i], counts[i] = v, n
	}
	return vecs, counts, nil
}

// queryDoc maps aikit's isQuery flag onto the checkpoint's own query and document prompts.
func queryDoc(isQuery bool) string {
	if isQuery {
		return "query"
	}
	return "document"
}

// Encode implements aikit's encoder.Encoder: the "query" prompt when isQuery, else the "document" prompt.
func (e *Encoder) Encode(text string, isQuery bool) ([]float32, error) {
	v, _, err := e.EmbedText(text, queryDoc(isQuery))
	return v, err
}

// EncodeBatch implements aikit's encoder.Encoder. concurrency is ignored: the matmuls parallelise internally.
func (e *Encoder) EncodeBatch(texts []string, isQueries []bool, concurrency int) ([][]float32, error) {
	vecs, _, err := e.EncodeBatchCounted(texts, isQueries, concurrency)
	return vecs, err
}

// EncodeBatchCounted is EncodeBatch with each input's token count.
func (e *Encoder) EncodeBatchCounted(texts []string, isQueries []bool, _ int) ([][]float32, []int, error) {
	vecs := make([][]float32, len(texts))
	counts := make([]int, len(texts))
	for i, t := range texts {
		q := i < len(isQueries) && isQueries[i]
		v, n, err := e.EmbedText(t, queryDoc(q))
		if err != nil {
			return nil, nil, fmt.Errorf("input %d: %w", i, err)
		}
		vecs[i], counts[i] = v, n
	}
	return vecs, counts, nil
}

// CountTokens is the token count Encode would see.
func (e *Encoder) CountTokens(text string, isQuery bool) int {
	ids, err := e.Tokenize(text, queryDoc(isQuery))
	if err != nil {
		return 0
	}
	return len(ids)
}

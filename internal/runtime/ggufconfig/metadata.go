package ggufconfig

import (
	"fmt"

	"github.com/townsendmerino/aikit/embed"
)

// ggufVocabSize reads the vocab size from the embedding tensor's dims (no
// dequant) or, failing that, the tokens metadata array length.
func ggufVocabSize(g *embed.GGUFFile) int {
	if dims, ok := g.Dims("token_embd.weight"); ok && len(dims) == 2 {
		return dims[1] // [in=hidden, out=vocab]
	}
	if toks, ok := g.Metadata["tokenizer.ggml.tokens"].([]any); ok {
		return len(toks)
	}
	return 0
}

// ggufEOS sets cfg.EOSTokenID from the GGUF tokenizer metadata (resolveEOSIDs
// falls back to this when there's no generation_config.json next to the .gguf).
func ggufEOS(g *embed.GGUFFile, cfg *Config) {
	if eos, ok := g.Uint("tokenizer.ggml.eos_token_id"); ok {
		cfg.EOSTokenID = fmt.Appendf(nil, "%d", eos)
	}
}

// ggufUintReader reads the unsigned metadata keys under prefix (a family's "arch." keys) as int, 0 when absent.
func ggufUintReader(g *embed.GGUFFile, prefix string) func(k string) int {
	return func(k string) int {
		v, _ := g.Uint(prefix + k)
		return int(v)
	}
}

// ggufFloatReader reads a float metadata key under prefix, or an unsigned one as float (a value some writers store as an
// integer), 0 when absent.
func ggufFloatReader(g *embed.GGUFFile, prefix string) func(k string) float64 {
	return func(k string) float64 {
		if v, ok := g.Float(prefix + k); ok {
			return v
		}
		v, _ := g.Uint(prefix + k)
		return float64(v)
	}
}

// ggufIntArray / ggufBoolArray coerce a GGUF array metadatum (parsed as []any)
// into a typed slice — for gemma4's per-layer feed_forward_length and the
// sliding/global bool pattern.
func ggufIntArray(v any) []int {
	a, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]int, len(a))
	for i, e := range a {
		switch n := e.(type) {
		case int:
			out[i] = n
		case int32:
			out[i] = int(n)
		case int64:
			out[i] = int(n)
		case uint32:
			out[i] = int(n)
		case uint64:
			out[i] = int(n)
		case float64:
			out[i] = int(n)
		}
	}
	return out
}

func ggufBoolArray(v any) []bool {
	a, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]bool, len(a))
	for i, e := range a {
		if b, ok := e.(bool); ok {
			out[i] = b
		}
	}
	return out
}

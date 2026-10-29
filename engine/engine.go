package engine

import (
	"context"

	"github.com/townsendmerino/goinfer/constrain"
	"github.com/townsendmerino/goinfer/decoder"
)

// Request is what an Engine needs to start a generation: the prompt, the budget, the sampling, an optional grammar
// constraint, and the KV scope.
type Request struct {
	PromptIDs []int
	MaxTokens int
	Sampling  decoder.SamplingParams
	// Masker is the request's grammar constraint (response_format or a tool grammar), nil when unconstrained.
	Masker *constrain.Masker
	// PlainConstrained asks for plain constrained decoding, never a speculative path (goinfer_confidence records one
	// position per emitted token).
	PlainConstrained bool
	Scope            decoder.KVScope // the request's KV sharing scope, handed to the decoder as it is
}

// Engine starts generations: Start begins one over req and streams its token ids. report gives the generation's
// account as it stands when called; the caller reads it once tokens has closed, and both it and its result may be nil. done is called once, after the stream has been read to its end.
type Engine interface {
	Start(ctx context.Context, req Request) (tokens <-chan int, report func() *Report, done func())
}

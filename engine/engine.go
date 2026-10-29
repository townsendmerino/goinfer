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
	// NoSpeculation asks for plain decoding on a CPU session, never a speculative path there (goinfer_confidence
	// records one position per emitted token). The resident path never speculates a constrained request.
	NoSpeculation bool
	Scope         decoder.KVScope // the request's KV sharing scope, handed to the decoder as it is
}

// Engine starts generations: Start begins one over req and returns its Stream. Cancel it through ctx; an error arrives
// in the stream's Report.
type Engine interface {
	Start(ctx context.Context, req Request) *Stream
}

package decoder

import (
	"context"

	"github.com/townsendmerino/goinfer/internal/runtime/residency"
)

type KVScope = residency.KVScope

// WithKVScope returns ctx carrying the request's KV sharing scope, which Generate and its variants read.
func WithKVScope(ctx context.Context, s KVScope) context.Context {
	return residency.ContextWithKVScope(ctx, s)
}

func kvScopeOf(ctx context.Context) KVScope { return residency.KVScopeOf(ctx) }

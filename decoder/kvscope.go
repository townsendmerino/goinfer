package decoder

import "context"

// KVScope is the sharing scope of one request's KV (docs/tasks/task-kv-architecture-2026-10.md §5.15). A request
// takes rows another sequence computed only from a sequence in the same Namespace, and under Off only by continuing a
// slot's whole sequence. The zero value is the single local namespace, sharing on.
type KVScope struct {
	Namespace string // "" is the single local namespace; serve sets one per API key
	Off       bool   // no rows from another sequence: a continuation of a slot's sequence only
}

type kvScopeKey struct{}

// WithKVScope returns ctx carrying the request's KV sharing scope, which Generate and its variants read.
func WithKVScope(ctx context.Context, s KVScope) context.Context {
	return context.WithValue(ctx, kvScopeKey{}, s)
}

// kvScopeOf is the scope ctx carries, or the zero scope.
func kvScopeOf(ctx context.Context) KVScope {
	if ctx == nil {
		return KVScope{}
	}
	s, _ := ctx.Value(kvScopeKey{}).(KVScope)
	return s
}

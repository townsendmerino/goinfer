package decoder

import "context"

// KVScope is the sharing scope of one request's KV (docs/tasks/task-kv-architecture-2026-10.md §5.15). A request
// takes rows another sequence computed only from a sequence in the same Namespace, and under Off only by continuing a
// slot's whole sequence. The zero value is the single local namespace, sharing on.
type KVScope struct {
	Namespace string // "" is the single local namespace; serve sets one per API key
	Off       bool   // no rows from another sequence: a continuation of a slot's sequence only
	// Boundary is how many leading ids of the request's prompt are the stable system-and-tools section, the part before the first conversation turn
	// (chat.Template.PrefixBoundary), 0 when the route does not know. A prefix snapshot is taken there during the cold prefill when the part of the
	// prompt after it is longer than a windowed layer's rollback (docs/tasks/task-kv7-step2-boundary-snapshot-2026-10.md).
	Boundary int
	// Keep is how many leading ids of the prompt the next turn's prompt repeats: the end of the prompt less its generation tail, the part a
	// chat template's history render rewrites (chat.NextTurns). A recurrent model's state is kept there, so the next turn continues from it
	// (docs/queue-engineering.md, B35). 0 when the route does not know.
	Keep int
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

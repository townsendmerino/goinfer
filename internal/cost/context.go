package cost

import "context"

type ctxKey struct{}

// WithCounter is ctx carrying c for one generation (SB11b: a request that asks for its cost). A nil c leaves ctx as it is,
// so a request that does not ask carries no counter and the decoder pays one context lookup per generation.
func WithCounter(ctx context.Context, c *Counter) context.Context {
	if c == nil {
		return ctx
	}
	return context.WithValue(ctx, ctxKey{}, c)
}

// FromContext is the counter ctx carries, nil (off) when it carries none.
func FromContext(ctx context.Context) *Counter {
	c, _ := ctx.Value(ctxKey{}).(*Counter)
	return c
}

package cost

import (
	"context"
	"testing"
)

// TestContext_carriesOnlyWhatWasAsked (G-SB11b-2's context half): a nil counter leaves the context unchanged, so a request that
// does not ask carries none; a counter put on a context comes back from it and from a context derived from it.
func TestContext_carriesOnlyWhatWasAsked(t *testing.T) {
	base := context.Background()
	if got := WithCounter(base, nil); got != base {
		t.Fatal("WithCounter(nil) wrapped the context")
	}
	if FromContext(base) != nil {
		t.Fatal("a context that carries no counter returned one")
	}
	c := New()
	ctx, cancel := context.WithCancel(WithCounter(base, c))
	defer cancel()
	if FromContext(ctx) != c {
		t.Fatal("the counter did not survive a derived context")
	}
	if FromContext(Off(ctx)) != nil || FromContext(ctx) != c {
		t.Fatal("Off did not turn the counter off for its context alone")
	}
	if Off(base) != base {
		t.Fatal("Off wrapped a context that carries no counter")
	}
}

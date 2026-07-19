package main

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

// TestGenErr gates M1's error-surfacing filter: a real terminal error must reach
// the client (500 / error SSE), while context.Canceled — our own stop-string
// cancel and client disconnects — is a clean end and must be swallowed.
func TestGenErr(t *testing.T) {
	boom := errors.New("backend forward failed")
	for _, c := range []struct {
		name string
		in   error
		want error
	}{
		{"clean end", nil, nil},
		{"stop-string / client cancel", context.Canceled, nil},
		{"wrapped cancel", fmt.Errorf("generate: %w", context.Canceled), nil},
		{"real backend error", boom, boom},
		{"deadline is a real error", context.DeadlineExceeded, context.DeadlineExceeded},
	} {
		if got := genErr(c.in); got != c.want {
			t.Errorf("%s: genErr(%v) = %v, want %v", c.name, c.in, got, c.want)
		}
	}
}

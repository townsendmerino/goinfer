package server

import (
	"strings"
	"testing"
)

// The flags are the binary's, parsed by its flag set: a listener flag is refused by name, an unknown one by the
// parser, and no model is an error. The handler itself is exercised in internal/serveapp, on a model.
func TestNewHandler_refusals(t *testing.T) {
	for _, c := range []struct {
		flags []string
		want  string
	}{
		{[]string{"-model", "x.gguf", "-addr", ":8080"}, "-addr configure goinfer-serve's listener"},
		{[]string{"-model", "x.gguf", "-web"}, "-web configure goinfer-serve's listener"},
		{[]string{"-no-such-flag"}, "flag provided but not defined"},
		{[]string{"x.gguf"}, "unrecognized argument"},
		{nil, "need at least one of -model and -embed-model"},
		{[]string{"-model", "x.gguf", "-thinking", "sometimes"}, "-thinking"},
	} {
		h, closer, err := NewHandler(c.flags...)
		if err == nil || !strings.Contains(err.Error(), c.want) || h != nil || closer != nil {
			t.Errorf("NewHandler(%q) = %v, %v, %v; want an error containing %q", c.flags, h, closer, err, c.want)
		}
	}
}

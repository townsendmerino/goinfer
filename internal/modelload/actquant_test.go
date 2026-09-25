package modelload

import (
	"strings"
	"testing"
)

// TestActivationSafeQuant: a Phi-3 source (queue-engineering.md H2) loaded at the default int4 gets
// weight-only int8, since per-row int8 activations round its outliers to zero; an explicit --quant
// is honoured with a warning; a family with no known hazard, or a precision that keeps activations
// in f32, is left alone. The Phi-3 source is the committed tiny fixture (config.json model_type
// "phi3"), so this runs without a real checkpoint.
func TestActivationSafeQuant(t *testing.T) {
	const phi3 = "../../testdata/phi3-tiny"
	for _, c := range []struct {
		src, quant, explicit, want, msg string
	}{
		{phi3, "int4", "", "int8", "note: loading at --quant int8"},
		{phi3, "int8int8", "", "int8", "note: loading at --quant int8"},
		{phi3, "int4", "int4", "int4", "warning: --quant int4"},
		{phi3, "int8", "", "int8", ""},
		{phi3, "f32", "", "f32", ""},
		{"../../testdata/nonexistent-model-dir", "int4", "", "int4", ""},
	} {
		got, msg := activationSafeQuant(c.src, c.quant, c.explicit)
		if got != c.want || !strings.HasPrefix(msg, c.msg) || (c.msg == "") != (msg == "") {
			t.Errorf("activationSafeQuant(%s, %q, explicit %q) = %q, %q; want %q, prefix %q", c.src, c.quant, c.explicit, got, msg, c.want, c.msg)
		}
	}
}

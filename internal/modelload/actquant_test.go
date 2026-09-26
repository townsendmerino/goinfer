package modelload

import (
	"strings"
	"testing"
)

// TestActivationSafeQuant: a Phi-3 source (queue-engineering.md H2) loaded at the default int4 gets
// int8int8 with per-32 activation scales, the configuration that passed the quality gate; an explicit
// int8int8 gets per-32 added; an explicit int4 is honoured with a warning; weight-only int8 / f32 and a
// family with no known hazard are left alone. The Phi-3 source is the committed tiny fixture
// (config.json model_type "phi3"), so this runs without a real checkpoint.
func TestActivationSafeQuant(t *testing.T) {
	const phi3 = "../../testdata/phi3-tiny"
	for _, c := range []struct {
		src, quant, explicit string
		wantQ                string
		wantG                int
		msg                  string
	}{
		{phi3, "int4", "", "int8int8", 32, "note: loading at --quant int8int8 with per-32"},
		{phi3, "int8int8", "int8int8", "int8int8", 32, "note: using per-32"},
		{phi3, "int4", "int4", "int4", 0, "warning: --quant int4"},
		{phi3, "int8", "", "int8", 0, ""},
		{phi3, "f32", "", "f32", 0, ""},
		{"../../testdata/nonexistent-model-dir", "int4", "", "int4", 0, ""},
	} {
		q, g, msg := activationSafeQuant(c.src, c.quant, 0, c.explicit)
		if q != c.wantQ || g != c.wantG || !strings.HasPrefix(msg, c.msg) || (c.msg == "") != (msg == "") {
			t.Errorf("activationSafeQuant(%s, %q, explicit %q) = %q, %d, %q; want %q, %d, prefix %q", c.src, c.quant, c.explicit, q, g, msg, c.wantQ, c.wantG, c.msg)
		}
	}
}

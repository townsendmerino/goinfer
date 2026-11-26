package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/families"
	"testing"
)

// GPT-2's activation_function must select the function it NAMES. ValidateGPT2 accepts both "gelu_new" and "gelu", so
// gpt2Architecture must not hardcode ActGeluTanh: a checkpoint declaring HF's exact "gelu" would silently run the tanh
// approximation. In HF these are different entries (ACT2FN["gelu"] is the exact erf, ACT2FN["gelu_new"] the tanh form),
// not two spellings of one function. The next GPT-2-family checkpoint that declares "gelu" must get the function it
// asked for.
func TestGPT2_activationSelectsTheNamedFunction(t *testing.T) {
	for _, c := range []struct {
		name string
		want ActKind
	}{
		{"", darch.ActGeluTanh},         // GPT-2's own default IS gelu_new
		{"gelu_new", darch.ActGeluTanh}, // tanh approximation
		{"gelu", darch.ActGelu},         // exact erf — the case that was wrong
	} {
		if got := families.GPT2Act(c.name); got != c.want {
			t.Errorf("families.GPT2Act(%q) = %v, want %v", c.name, got, c.want)
		}
	}
}

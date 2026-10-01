package decide

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func TestHeadQuant(t *testing.T) {
	for _, tc := range []struct {
		chosen string
		set    bool
		want   string
	}{
		{"", false, decoder.DecisionHeadQuant},     // nothing chosen: the decision models' default
		{"int4", false, decoder.DecisionHeadQuant}, // a process default that was not passed is not a choice
		{"int4", true, "int4"},
		{"f32", true, "f32"},
		{"", true, ""}, // an explicit "" is f32 (serve's quant= with no value), not "unset"
	} {
		if got := HeadQuant(tc.chosen, tc.set); got != tc.want {
			t.Errorf("HeadQuant(%q, %v) = %q, want %q", tc.chosen, tc.set, got, tc.want)
		}
	}
	if decoder.DecisionHeadQuant != "int8int8" {
		t.Errorf("DecisionHeadQuant = %q; D6b's decision (2026-10-01) is int8int8, so a change needs its own record", decoder.DecisionHeadQuant)
	}
}

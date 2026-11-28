package wbuild

import "testing"

// TestMatmulQuant_routerStaysInt8UnderInt4Mix pins that QuantInt4Mix keeps the router (ffn_gate_inp*) at int8, as its doc
// comment promises, while the FFN bulk goes to int4: a match on "ffn_" alone would take the router with the bulk. Every
// current load site routes the router through streamMat(..., QuantNone, ...) and never calls MatmulQuant, so this guards a
// future family that reaches the router through the generic mat() helper.
func TestMatmulQuant_routerStaysInt8UnderInt4Mix(t *testing.T) {
	for _, name := range []string{
		"blk.0.ffn_gate_inp.weight",
		"blk.3.ffn_gate_inp.weight",
	} {
		if got := MatmulQuant(QuantInt4Mix, name); got != QuantInt8 {
			t.Errorf("MatmulQuant(QuantInt4Mix, %q) = %v, want QuantInt8 (the router)", name, got)
		}
	}
	// Sanity: the FFN bulk this guardrail must NOT catch stays int4.
	for _, name := range []string{
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_gate_exps.weight",
		"blk.0.ffn_up_exps.weight",
		"blk.0.ffn_down_exps.weight",
	} {
		if got := MatmulQuant(QuantInt4Mix, name); got != QuantInt4 {
			t.Errorf("MatmulQuant(QuantInt4Mix, %q) = %v, want QuantInt4 (the FFN bulk)", name, got)
		}
	}
}

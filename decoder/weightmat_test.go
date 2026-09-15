package decoder

import "testing"

// TestMatmulQuant_routerStaysInt8UnderInt4Mix gates M-27's guardrail: matmulQuant keyed
// quantInt4Mix's "FFN bulk" branch on any tensor name containing "ffn_", which also matches
// the router (ffn_gate_inp*) — the opposite of quantInt4Mix's own doc comment, which promises
// the router stays at int8 alongside attention. Every current load site already routes the
// router through streamMat(..., quantNone, ...) directly and never calls matmulQuant at all, so
// this guards against a future family reaching the router through the generic mat() helper and
// repeating that mistake.
func TestMatmulQuant_routerStaysInt8UnderInt4Mix(t *testing.T) {
	for _, name := range []string{
		"blk.0.ffn_gate_inp.weight",
		"blk.3.ffn_gate_inp.weight",
	} {
		if got := matmulQuant(quantInt4Mix, name); got != quantInt8 {
			t.Errorf("matmulQuant(quantInt4Mix, %q) = %v, want quantInt8 (the router)", name, got)
		}
	}
	// Sanity: the FFN bulk this guardrail must NOT catch stays int4.
	for _, name := range []string{
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_gate_exps.weight",
		"blk.0.ffn_up_exps.weight",
		"blk.0.ffn_down_exps.weight",
	} {
		if got := matmulQuant(quantInt4Mix, name); got != quantInt4 {
			t.Errorf("matmulQuant(quantInt4Mix, %q) = %v, want quantInt4 (the FFN bulk)", name, got)
		}
	}
}

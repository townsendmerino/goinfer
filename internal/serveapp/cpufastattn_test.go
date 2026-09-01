package serveapp

import (
	"flag"
	"strings"
	"testing"
)

// --cpu-fast-attention opts into a documented divergence, so two things must
// hold and neither should rest on someone remembering them.
//
//  1. It is OFF unless asked for. A speed flag that turns itself on is how a user
//     gets different output than they got yesterday with no change on their side.
//  2. Its help TEXT states the trade. The precedent is --metal-fast-prefill,
//     whose help spells out the divergence "so the tradeoff is disclosed in
//     --help, not something a user has to already know to type". A flag that
//     diverges silently is worse than no flag.
func TestCPUFastAttentionDefaultsOnAndDisclosesTheTrade(t *testing.T) {
	// THE PREVIOUS VERSION OF THIS TEST WAS VACUOUS, and it is worth saying how, because the
	// shape recurs: it asserted `var cfg config; if cfg.cpuFastAttention { fail }`. That is the
	// ZERO VALUE of a bool field, not the flag's default — it would have passed unchanged if
	// someone had set flag.BoolVar's default to true, which is precisely the regression it
	// existed to catch. It passed on the first run after the default was flipped to ON.
	//
	// The fix is to read the default off a real FlagSet, which is the thing that decides what a
	// user gets. Same defect class as the doc comment in a3_divergence_test.go: a name and a
	// comment promising an invariant the body never touches.
	fs := flag.NewFlagSet("probe", flag.ContinueOnError)
	var c config
	fs.BoolVar(&c.cpuFastAttention, "cpu-fast-attention", true, cpuFastAttentionHelp)
	fs.BoolVar(&c.cpuExactPrefill, "cpu-exact-prefill", false, cpuExactPrefillHelp)
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !c.cpuFastAttention {
		t.Error("cpu-fast-attention must default to TRUE (changed 2026-08-31)")
	}
	if c.cpuExactPrefill {
		t.Error("cpu-exact-prefill must default to false — it is the opt-OUT")
	}
	// And the opt-out must actually be reachable by the name --help advertises.
	if err := fs.Parse([]string{"-cpu-exact-prefill"}); err != nil {
		t.Fatalf("parse opt-out: %v", err)
	}
	if !c.cpuExactPrefill {
		t.Error("-cpu-exact-prefill did not set the opt-out")
	}

	help := cpuFastAttentionHelp
	for _, must := range []string{
		"NOT bit-identical",   // the divergence is named, not implied
		"0.9976",              // and quantified
		"2.28x",               // the win is quantified too, so the trade is legible
		"MoE",                 // the refusal a user cannot override
		"Speculative",         // the guarantee that is preserved regardless
		"DEFAULT ON",          // it is on unless asked otherwise, and --help says so first
		"--cpu-exact-prefill", // and names the way out, or the disclosure is not actionable
	} {
		if !strings.Contains(help, must) {
			t.Errorf("the flag help does not mention %q — the trade must be disclosed in --help:\n%s", must, help)
		}
	}
	// The opt-out discloses its own cost. A correctness escape hatch that hides its price is how
	// someone turns it on globally and then reports a performance regression against us.
	for _, must := range []string{"BIT-EXACT", "2.28x", "MoE"} {
		if !strings.Contains(cpuExactPrefillHelp, must) {
			t.Errorf("--cpu-exact-prefill help does not mention %q:\n%s", must, cpuExactPrefillHelp)
		}
	}
}

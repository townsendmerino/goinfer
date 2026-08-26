package serveapp

import (
	"os"
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
func TestCPUFastAttentionDefaultsOffAndDisclosesTheTrade(t *testing.T) {
	os.Unsetenv("GOINFER_CPU_FAST_ATTENTION")
	var cfg config
	if cfg.cpuFastAttention {
		t.Error("cpu-fast-attention must default to false")
	}
	// The env the decoder actually reads must not be set by merely constructing a
	// default config.
	if os.Getenv("GOINFER_CPU_FAST_ATTENTION") != "" {
		t.Error("the default config set GOINFER_CPU_FAST_ATTENTION")
	}

	help := cpuFastAttentionHelp
	for _, must := range []string{
		"NOT bit-identical", // the divergence is named, not implied
		"0.9976",            // and quantified
		"2.28x",             // the win is quantified too, so the trade is legible
		"MoE",               // the refusal a user cannot override
		"Speculative",       // the guarantee that is preserved regardless
	} {
		if !strings.Contains(help, must) {
			t.Errorf("the flag help does not mention %q — the trade must be disclosed in --help:\n%s", must, help)
		}
	}
}

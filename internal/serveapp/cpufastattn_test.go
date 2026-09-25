package serveapp

import (
	"flag"
	"strings"
	"testing"
)

// The CPU's f32 prompt attention is ON by default and is a documented divergence, so its opt-out's help has to
// disclose the trade — a divergence "disclosed in --help, not something a user has to already know to type".
// Since --cpu-fast-attention was removed (it defaulted to true, so its only reachable use, =false, was
// --cpu-exact-prefill under another name), --cpu-exact-prefill's help is the ONLY place --help says any of this,
// so it has to carry all of it.
func TestCPUExactPrefillDisclosesTheDefaultsTrade(t *testing.T) {
	// Read the default off a real FlagSet, not the zero value of a config field: a zero-value check passes
	// unchanged if someone flips the flag's default, which is exactly the regression such a check exists for.
	fs := flag.NewFlagSet("probe", flag.ContinueOnError)
	var c config
	fs.BoolVar(&c.cpuExactPrefill, "cpu-exact-prefill", false, cpuExactPrefillHelp)
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.cpuExactPrefill {
		t.Error("cpu-exact-prefill must default to false — it is the opt-OUT; the fast path is the default")
	}
	if err := fs.Parse([]string{"-cpu-exact-prefill"}); err != nil {
		t.Fatalf("parse opt-out: %v", err)
	}
	if !c.cpuExactPrefill {
		t.Error("-cpu-exact-prefill did not set the opt-out")
	}
	for _, must := range []string{
		"BIT-EXACT",         // what the flag buys
		"DEFAULT",           // that the other path is the default, not an opt-in
		"NOT bit-identical", // the default's divergence is named, not implied
		"0.9976",            // and quantified
		"2.28x",             // and so is the win, so the trade is legible
		"512",               // the floor below which the default runs the exact kernel anyway
		"MoE",               // MoE takes the same path — so this is its only exact route too
		"speculative",       // the guarantee that holds regardless
		"--exact-prefill",   // and the all-backends version
	} {
		if !strings.Contains(cpuExactPrefillHelp, must) {
			t.Errorf("--cpu-exact-prefill help does not mention %q — the default's trade must be disclosed in --help:\n%s", must, cpuExactPrefillHelp)
		}
	}
	// --exact-prefill's help quotes each backend's floor; Metal's is 64 (metal/backend.go metalFastPrefillFloor),
	// which the help used to give as 512 (and the removed --metal-fast-prefill's as 256).
	if !strings.Contains(exactPrefillHelp, "above 64 prompt tokens") {
		t.Errorf("--exact-prefill help does not give Metal's 64-token floor:\n%s", exactPrefillHelp)
	}
	for _, gone := range []string{"--cpu-fast-attention", "--metal-fast-prefill"} {
		if strings.Contains(exactPrefillHelp, gone) || strings.Contains(cpuExactPrefillHelp, gone) {
			t.Errorf("help text still names the removed flag %s", gone)
		}
	}
}

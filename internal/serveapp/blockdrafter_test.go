package serveapp

import (
	"strings"
	"testing"
)

// TestAttachBlockDrafter_declinesWithoutResident gates the startup behaviour an operator
// actually depends on: --drafter on a backend that cannot host one FAILS, with a reason.
//
// The alternative — attaching silently and serving at 1x — is the worse outcome by far: block
// drafting is invisible when it does not engage, because output is identical either way
// (it is lossless by construction). An operator would see correct responses at plain speed and
// have nothing to look at. So the decline is the feature.
func TestAttachBlockDrafter_declinesWithoutResident(t *testing.T) {
	lm := &loadedModel{name: "test"} // no model ⇒ not block-spec capable
	err := attachBlockDrafter(lm, "/nonexistent")
	if err == nil {
		t.Fatal("attachBlockDrafter succeeded with no resident model — it must decline")
	}
	if !strings.Contains(err.Error(), "resident") {
		t.Errorf("decline should name the resident requirement, got: %v", err)
	}
	if lm.blockSpec != nil {
		t.Error("blockSpec set despite the decline")
	}
}

// TestWarnThinkingTemplate covers the branch, since the warning is the only signal an operator
// gets for a configuration that is otherwise completely silent — correct output, reduced speed,
// nothing in any log.
func TestWarnThinkingTemplate_noTokenizer(t *testing.T) {
	warnThinkingTemplate(&loadedModel{name: "x"}) // nil tokenizer must not panic
}

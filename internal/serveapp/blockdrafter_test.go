package serveapp

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

// TestAttachBlockDrafter_declinesWithoutResident gates the startup behaviour an operator
// actually depends on: --drafter on a backend that cannot host one FAILS, with a reason.
//
// The alternative — attaching silently and serving at 1x — is the worse outcome by far: block
// drafting is invisible when it does not engage, because output is identical either way
// (it is lossless by construction). An operator would see correct responses at plain speed and
// have nothing to look at. So the decline is the feature.
func TestAttachBlockDrafter_declinesWithoutResident(t *testing.T) {
	lm := &loadedModel{name: "test"}      // no model ⇒ not block-spec capable
	err := attachBlockDrafter(lm, nil, 0) // never dereferenced: the resident check declines first
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

// TestWarnThinkingTemplate: the warning follows the served template's thinking, not the vocabulary: a drafter beside a template
// whose replies can think warns; -thinking off, a checkpoint whose own default is off, no template, and no drafter do not.
func TestWarnThinkingTemplate(t *testing.T) {
	nine := templateFromGolden(t, "qwen3.5-9b") // its template opens a think block by default
	if !nine.WithThinking(chat.ThinkTemplate).ThinkingPossible() {
		t.Fatal("qwen3.5-9b's golden template no longer thinks by default; pick a fixture that does")
	}
	bs := &fakeDrafter{}
	for _, c := range []struct {
		name string
		lm   *loadedModel
		want bool
	}{
		{"drafter, template default on", &loadedModel{name: "x", blockSpec: bs, tmpl: nine.WithThinking(chat.ThinkTemplate)}, true},
		{"drafter, -thinking on", &loadedModel{name: "x", blockSpec: bs, tmpl: nine.WithThinking(chat.ThinkOn)}, true},
		{"drafter, -thinking off", &loadedModel{name: "x", blockSpec: bs, tmpl: nine.WithThinking(chat.ThinkOff)}, false},
		{"drafter, no template", &loadedModel{name: "x", blockSpec: bs}, false},
		{"no drafter", &loadedModel{name: "x", tmpl: nine.WithThinking(chat.ThinkTemplate)}, false},
	} {
		if got := warnThinkingTemplate(c.lm); got != c.want {
			t.Errorf("%s: warned %v, want %v", c.name, got, c.want)
		}
	}
}

// TestDrafterWidth_validate: -drafter-width without -drafter is a startup error, not silently ignored, and so are 1 and negative
// widths; 0 and widths of 2 or more pass validation (the attach checks a width against the drafter and the target).
func TestDrafterWidth_validate(t *testing.T) {
	for _, c := range []struct {
		drafter string
		width   int
		ok      bool
	}{
		{"", 0, true}, {"", 4, false}, {"", -1, false},
		{"d", 0, true}, {"d", 4, true}, {"d", 16, true}, {"d", 1, false}, {"d", 2, false}, {"d", 3, false}, {"d", -1, false},
	} {
		// A bare config fails other required flags; only whether a -drafter-width refusal appears is this test's.
		err := (config{drafter: c.drafter, drafterWidth: c.width}).validate()
		if refused := err != nil && strings.Contains(err.Error(), "-drafter-width"); refused == c.ok {
			t.Errorf("-drafter %q -drafter-width %d: err %v, want a -drafter-width refusal %v", c.drafter, c.width, err, !c.ok)
		}
	}
}

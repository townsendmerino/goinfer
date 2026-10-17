package serveapp

import (
	"encoding/json"
	"flag"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/modelload"
)

// KV7 step 2, M3 (docs/tasks/task-kv7-step2-boundary-snapshot-2026-10.md §8): the boundary is where the stable system-and-tools section ends, read from the
// render, and the same for two sessions of one client whatever their first message says.
func TestPrefixBoundary_endsWhereTheFirstTurnBegins(t *testing.T) {
	tk, err := modelload.Tokenizer(filepath.Join("..", "..", "testdata", "mellum2-tokenizer"))
	if err != nil {
		t.Skipf("needs the Mellum2 tokenizer: %v", err)
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	sf := registerFlags(fs)
	_ = fs.Parse(nil)
	s := &server{cfg: sf.cfg}
	lm := &loadedModel{tk: tk, name: "mellum2.1-thinking"}
	lm.tmpl = s.cfg.tuneTemplate(mellum21Template(t))
	tm := lm.tmpl
	system := "You are a coding agent. " + strings.Repeat("Follow the instructions in the tools. ", 40)
	tools := []chat.Tool{{Name: "Read", Description: "Read a file. " + strings.Repeat("Paths are absolute. ", 30), Parameters: json.RawMessage(`{"type":"object"}`)},
		{Name: "Bash", Description: "Run a command.", Parameters: json.RawMessage(`{"type":"object"}`)}}
	render := func(user string) ([]int, int) {
		turns := []chat.Turn{{Role: "user", Content: user}}
		ids, err := lm.tk.EncodeSegments(tm.RenderToolsSegments(system, turns, tools), false)
		if err != nil {
			t.Fatal(err)
		}
		return ids, lm.prefixBoundaryOf(tm, system, turns, tools, ids)
	}
	idsA, bA := render("What does Add(2, 3) return? " + strings.Repeat("Think step by step. ", 30))
	idsB, bB := render("Rename sum to total everywhere.")
	if bA <= 0 || bA != bB {
		t.Fatalf("boundaries %d and %d: want one positive number for two sessions of one client", bA, bB)
	}
	// it is a prefix of both, shorter than either, and the prompts agree up to it
	if bA >= len(idsA) || bB >= len(idsB) || idsLCP(idsA, idsB) < bA {
		t.Fatalf("boundary %d, prompts of %d and %d ids sharing %d", bA, len(idsA), len(idsB), idsLCP(idsA, idsB))
	}
	// and it stops at the first turn's opening marker: the ids just before it end with the turn-open token, the role and content come after
	if got := renderText(lm, idsA[bA-1:bA]); got != "<|im_start|>" {
		t.Errorf("the id just before the boundary reads %q, want the first turn's opening marker", got)
	}
	// the system and tools section is inside it: a prompt with no turn content change renders the same boundary, and a longer system moves it
	long := system + strings.Repeat("More. ", 50)
	system2 := system
	system = long
	_, bLong := render("What does Add(2, 3) return?")
	system = system2
	if bLong <= bA {
		t.Errorf("a longer system prompt gave boundary %d, not past %d", bLong, bA)
	}
	// nothing to find without a turn or a template
	if lm.prefixBoundaryOf(tm, system, nil, tools, idsA) != 0 || lm.prefixBoundaryOf(nil, system, []chat.Turn{{Role: "user", Content: "x"}}, tools, idsA) != 0 {
		t.Error("a boundary was reported with no turn or no template")
	}
	// ids that are not the render's give none rather than a wrong number
	if got := lm.prefixBoundaryOf(tm, system, []chat.Turn{{Role: "user", Content: "x"}}, tools, idsA[1:]); got != 0 {
		t.Errorf("boundary %d for ids that do not start with the render's prefix", got)
	}
}

package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func loadGemma4Old(t *testing.T) histGolden {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "chat_think_goldens", "gemma4_old.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gs []histGolden
	if err := json.Unmarshal(raw, &gs); err != nil || len(gs) != 1 {
		t.Fatalf("golden: %v (%d entries)", err, len(gs))
	}
	return gs[0]
}

// The earlier Gemma 4 template (the E2B GGUF's) is managed, as its own variant: not the canonical spec, whose thinking-off
// prompt ends in a closed scaffold this template does not have.
func TestGemma4Old_detected(t *testing.T) {
	g := loadGemma4Old(t)
	tm, err := Detect(Meta{ChatTemplate: g.ChatTemplate})
	if err != nil {
		t.Fatal(err)
	}
	r := tm.Reasoning()
	if r == nil || !r.gemma4 || !r.oldGemma4 {
		t.Fatalf("the earlier template was not recognised as the earlier Gemma 4 variant: %+v", r)
	}
	if r.DefaultOn() || r.Open() != "<|channel>thought\n" || r.Close() != "<channel|>" {
		t.Fatalf("spec = %+v", r)
	}
	if tm.UsesNativeTools() {
		t.Fatal("the earlier template's tool prompt is not the canonical port; native tools must stay canonical-only")
	}
	// And the canonical one is still the canonical variant.
	c := loadHistGoldens(t)
	for _, e := range c {
		if !strings.HasPrefix(e.Checkpoint, "gemma-4") {
			continue
		}
		ct, _ := Detect(Meta{ChatTemplate: e.ChatTemplate})
		if cr := ct.Reasoning(); cr == nil || cr.oldGemma4 {
			t.Fatalf("canonical template misread as the earlier one: %+v", cr)
		}
	}
}

// TestGemma4Old_matchHF: every prompt the earlier template renders, byte for byte, in each thinking mode — declarations, the tool loop
// in progress (which keeps a calling turn's reasoning and does NOT reopen the channel after a result, unlike the canonical template)
// and a finished loop. Text beside a call is not in the cases: goinfer writes it before the call, the template after the results, and
// that order was only measured on the canonical template (docs/measurements/gemma4-tool-text-order-2026-09-30/).
func TestGemma4Old_matchHF(t *testing.T) {
	g := loadGemma4Old(t)
	tmpl, err := Detect(Meta{ChatTemplate: g.ChatTemplate})
	if err != nil {
		t.Fatal(err)
	}
	ran := 0
	for _, c := range g.Cases {
		turns := goldenTurns(t, c.Messages)
		system := ""
		if len(turns) > 0 && turns[0].Role == "system" {
			system, turns = turns[0].Content, turns[1:]
		}
		for mode, want := range c.Prompts {
			tm := tmpl.WithThinking(histMode[mode])
			var got string
			if c.Tools {
				got = tm.RenderTools(system, turns, goldenTools)
			} else {
				got = tm.Render(system, turns)
			}
			ran++
			if got != want {
				t.Errorf("%s / enable_thinking=%s:\n got %q\nwant %q", c.Name, mode, got, want)
			}
		}
	}
	if ran < 20 {
		t.Fatalf("only %d prompts compared", ran)
	}
}

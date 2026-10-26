//go:build goinfer_testhooks

package decoder

import (
	"strings"
	"testing"
)

// The G-FID prompt set as registered: N distinct gate prompts a cell, the pilot's two a cell, and no passage shared between the pilot
// and the gate.
func TestKSO4_promptSetShape(t *testing.T) {
	gate, pilot := ksO4PromptSet("gate", 12), ksO4PromptSet("pilot", 0)
	seen := map[string]bool{}
	count := map[string]int{}
	for _, p := range gate {
		if seen[p.text] {
			t.Fatalf("gate prompt %s repeats another's text", p.id)
		}
		seen[p.text] = true
		count[p.cell]++
		for _, pt := range ksO4PilotTexts {
			if strings.Contains(p.text, ksO4FirstSentences(pt, 1)) {
				t.Fatalf("gate prompt %s uses a pilot passage", p.id)
			}
		}
	}
	if count["short"] != 12 || count["long"] != 12 || len(pilot) != 4 {
		t.Fatalf("gate cells %v, pilot %d prompts; want 12 a cell and 4", count, len(pilot))
	}
	for _, p := range gate {
		if p.cell == "short" && strings.Count(p.text, ". ") > 3 {
			t.Fatalf("short prompt %s has more than two sentences of passage: %q", p.id, p.text)
		}
	}
}

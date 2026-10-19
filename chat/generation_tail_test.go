package chat

import (
	"strings"
	"testing"
)

// TestGenerationTail_realTemplates pins GenerationTail on the four checkpoints' real chat templates (testdata/chat_think_goldens), in
// every thinking setting. The named cases are the ones B35 rests on: Qwen3.5-9B with thinking off ends its prompt with an empty think
// block its history render drops (docs/measurements/qwen35-reuse-probe-2026-10-18/README.md), and with thinking on with an open one.
// For every case, a real next turn (an actual reply, then a new question) starts with the prompt minus the tail, so a state kept at
// that point is continued by the next request and not only by the probe.
func TestGenerationTail_realTemplates(t *testing.T) {
	want := map[string]string{
		"qwen3.5-9b false": "<think>\n\n</think>\n\n",
		"qwen3.5-9b true":  "<think>\n",
		"qwen3.5-9b unset": "<think>\n",
		"qwen3-4b true":    "",
	}
	seen := 0
	for _, g := range loadThinkGoldens(t) {
		tmpl, err := Detect(Meta{ChatTemplate: g.ChatTemplate})
		if err != nil {
			t.Fatalf("%s: %v", g.Checkpoint, err)
		}
		for mode, m := range goldenMode {
			tm := tmpl.WithThinking(m)
			for _, conv := range []string{"user", "sys_user", "multi"} {
				system, turns := convTurns(conv)
				gen := tm.Render(system, turns)
				tail := tm.GenerationTail(system, turns)
				if !strings.HasSuffix(gen, tail) {
					t.Fatalf("%s %s %s: the tail %q is not the end of the prompt", g.Checkpoint, mode, conv, tail)
				}
				if w, ok := want[g.Checkpoint+" "+mode]; ok {
					seen++
					if tail != w {
						t.Errorf("%s thinking=%s %s: tail %q, want %q", g.Checkpoint, mode, conv, tail, w)
					}
				}
				keep := gen[:len(gen)-len(tail)]
				next := tm.Render(system, append(append([]Turn(nil), turns...), Turn{Role: "assistant", Content: "Sure, here it is."},
					Turn{Role: "user", Content: "And the next part?"}))
				if !strings.HasPrefix(next, keep) {
					t.Errorf("%s thinking=%s %s: a real next turn does not start with the prompt minus its tail", g.Checkpoint, mode, conv)
				}
			}
		}
	}
	if seen != 4*3 {
		t.Fatalf("checked %d named cases, want 12", seen)
	}
}

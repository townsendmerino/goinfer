package streamframe

import (
	"encoding/json"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
)

func TestParseBudgetFlag(t *testing.T) {
	cases := []struct {
		in      string
		want    BudgetFlag
		wantErr bool
	}{
		{"", BudgetFlag{}, false}, {"auto", BudgetFlag{}, false}, {"AUTO", BudgetFlag{}, false},
		{"unlimited", BudgetFlag{Unlimited: true}, false}, {"512", BudgetFlag{N: 512}, false},
		{"0", BudgetFlag{}, true}, {"-3", BudgetFlag{}, true}, {"lots", BudgetFlag{}, true},
	}
	for _, tc := range cases {
		got, err := ParseBudgetFlag(tc.in)
		if (err != nil) != tc.wantErr || (err == nil && got != tc.want) {
			t.Errorf("ParseBudgetFlag(%q) = %+v, %v; want %+v, err=%v", tc.in, got, err, tc.want, tc.wantErr)
		}
	}
}

func TestResolveBudget(t *testing.T) {
	auto, unlimited, cap100 := BudgetFlag{}, BudgetFlag{Unlimited: true}, BudgetFlag{N: 100}
	cases := []struct {
		name      string
		f         BudgetFlag
		explicit  int
		maxTokens int
		want      int
		ok        bool
	}{
		{"auto: three quarters of a 160-token turn", auto, 0, 160, 120, true},
		{"auto: a 4096-token turn", auto, 0, 4096, 3072, true},
		{"auto: 8 tokens leaves 2 to answer", auto, 0, 8, 6, true},
		{"a turn under 4 tokens is left alone", auto, 0, 3, 0, false},
		{"request budget below the room holds", auto, 50, 160, 50, true},
		{"request budget above the room is clamped to it", auto, 1024, 1200, 900, true},
		{"Anthropic-style budget equal to max_tokens is clamped", auto, 1200, 1200, 900, true},
		{"unlimited applies no ceiling of its own", unlimited, 0, 4096, 0, false},
		{"unlimited still honours a request budget", unlimited, 200, 4096, 200, true},
		{"-reasoning-budget N caps an unasked request", cap100, 0, 4096, 100, true},
		{"N is itself clamped by the room", cap100, 0, 40, 30, true},
		{"a request budget overrides N", cap100, 300, 4096, 300, true},
	}
	for _, tc := range cases {
		got, ok := ResolveBudget(tc.f, tc.explicit, tc.maxTokens)
		if ok != tc.ok || (ok && got != tc.want) {
			t.Errorf("%s: got (%d, %v), want (%d, %v)", tc.name, got, ok, tc.want, tc.ok)
		}
	}
}

func TestResolveThink(t *testing.T) {
	ts, err := ResolveThink(chat.ThinkTemplate, FormatSplit, ThinkRequest{
		Kwargs:          map[string]json.RawMessage{"enable_thinking": json.RawMessage(`true`)},
		ReasoningEffort: "high",
		Format:          "deepseek-legacy",
		Budget:          128,
	})
	if err != nil {
		t.Fatalf("ResolveThink: %v", err)
	}
	if !ts.Explicit || ts.Mode != chat.ThinkOn || ts.Format != FormatLegacy || ts.Effort != "high" || ts.Budget != 128 {
		t.Fatalf("unexpected ThinkSettings: %+v", ts)
	}
	if _, err := ResolveThink(chat.ThinkTemplate, FormatSplit, ThinkRequest{Format: "bogus"}); err == nil {
		t.Error("expected error on unknown reasoning_format")
	}
	if _, err := ResolveThink(chat.ThinkTemplate, FormatSplit, ThinkRequest{Budget: -1}); err == nil {
		t.Error("expected error on negative budget")
	}
}

package chatapp

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
)

// quietly runs f with stdout sent to a file, since the commands print their confirmations there.
func quietly(t *testing.T, f func()) {
	t.Helper()
	out, err := os.Create(filepath.Join(t.TempDir(), "stdout"))
	if err != nil {
		t.Fatal(err)
	}
	prev := os.Stdout
	os.Stdout = out
	defer func() { os.Stdout = prev; out.Close() }()
	f()
}

// Each REPL command changes what it names and nothing else; a value it cannot parse leaves the setting as it was; only /quit
// and /exit end the loop.
func TestCommand_changesWhatItNames(t *testing.T) {
	base := func() *session {
		return &session{system: "sys", history: []msg{{"user", "u"}, {"assistant", "a"}}, maxTok: 100, think: chat.ThinkTemplate,
			sp: decoder.SamplingParams{Temperature: 0.7, TopK: 40, TopP: 0.9, Seed: 1}}
	}
	for _, c := range []struct {
		line  string
		quits bool
		want  func(*session) bool
	}{
		{"/quit", true, func(s *session) bool { return len(s.history) == 2 }},
		{"/exit", true, func(s *session) bool { return len(s.history) == 2 }},
		{"/reset", false, func(s *session) bool { return s.history == nil && s.system == "sys" }},
		{"/system be brief", false, func(s *session) bool { return s.system == "be brief" && s.history == nil }},
		{"/temp 0.2", false, func(s *session) bool { return s.sp.Temperature == 0.2 && s.sp.TopP == 0.9 }},
		{"/temp warm", false, func(s *session) bool { return s.sp.Temperature == 0.7 }},
		{"/topk 5", false, func(s *session) bool { return s.sp.TopK == 5 && s.sp.TopP == 0.9 }},
		{"/topk many", false, func(s *session) bool { return s.sp.TopK == 40 }},
		{"/topp 0.5", false, func(s *session) bool { return s.sp.TopP == 0.5 && s.sp.Temperature == 0.7 }},
		{"/seed 42", false, func(s *session) bool { return s.sp.Seed == 42 }},
		{"/max 7", false, func(s *session) bool { return s.maxTok == 7 && len(s.history) == 2 }},
		{"/max", false, func(s *session) bool { return s.maxTok == 100 }},
		{"/think off", false, func(s *session) bool { return s.think == chat.ThinkOff }},
		{"/think sideways", false, func(s *session) bool { return s.think == chat.ThinkTemplate }},
		{"/json", false, func(s *session) bool { return s.jsonOut }},
		{"/frobnicate", false, func(s *session) bool { return len(s.history) == 2 && s.system == "sys" && s.maxTok == 100 }},
		{"/demo no-such-demo", false, func(s *session) bool { return len(s.history) == 2 && s.system == "sys" }},
	} {
		s := base()
		var quits bool
		quietly(t, func() { quits = s.command(c.line) })
		if quits != c.quits || !c.want(s) {
			t.Errorf("%q: quits %v (want %v), session after: system %q, %d history, temp %v, topk %d, topp %v, seed %d, max %d, think %v, json %v",
				c.line, quits, c.quits, s.system, len(s.history), s.sp.Temperature, s.sp.TopK, s.sp.TopP, s.sp.Seed, s.maxTok, s.think, s.jsonOut)
		}
	}
	s := base()
	quietly(t, func() { s.command("/json"); s.command("/json") })
	if s.jsonOut {
		t.Error("/json twice left JSON output on: it toggles")
	}
}

// /demo runs its canned prompt as one turn against the model, the prompt and the reply joining the history, and puts the JSON
// setting back after a JSON demo, since the demo's setting is for its own turn.
func TestCommand_demoRunsOneTurnAndRestoresTheJSONSetting(t *testing.T) {
	var jsonDemo string
	for _, d := range demos {
		if d.json && d.system == "" {
			jsonDemo = d.name
			break
		}
	}
	if jsonDemo == "" {
		t.Fatal("no JSON demo without a system prompt to run")
	}
	s, err := loadFromPath(tinyChatModel(t), decoder.Options{Backend: "cpu"}, false, "")
	if err != nil {
		t.Fatal(err)
	}
	s.plain, s.maxTok, s.sp.Temperature = true, 4, 0
	quietly(t, func() { s.out = os.Stdout; s.command("/demo " + jsonDemo) })
	if len(s.history) != 2 || s.history[0].role != "user" || s.history[1].role != "assistant" || s.history[1].content == "" {
		t.Errorf("history after /demo: %+v, want the demo's prompt and a reply", s.history)
	}
	if s.jsonOut {
		t.Error("the JSON demo left JSON output on")
	}
}

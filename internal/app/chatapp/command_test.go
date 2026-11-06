package chatapp

import (
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
)

// quietly runs f with stdout discarded, since the commands print their confirmations there.
func quietly(t *testing.T, f func()) {
	t.Helper()
	out, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	prev := os.Stdout
	os.Stdout = out
	defer func() { os.Stdout = prev; out.Close() }()
	f()
}

// Each REPL command changes what it names and nothing else, held by comparing the whole session after it with the session before
// plus that one change; a value it cannot parse changes nothing; only /quit and /exit end the loop.
func TestCommand_changesWhatItNames(t *testing.T) {
	base := func() *session {
		return &session{system: "sys", history: []msg{{"user", "u"}, {"assistant", "a"}}, maxTok: 100, think: chat.ThinkTemplate,
			effort: "low", sp: decoder.SamplingParams{Temperature: 0.7, TopK: 40, TopP: 0.9, Seed: 1}}
	}
	none := func(*session) {}
	for _, c := range []struct {
		line   string
		quits  bool
		change func(*session) // the one change the command makes to base
	}{
		{"/quit", true, none},
		{"/exit", true, none},
		{"/reset", false, func(s *session) { s.history = nil }},
		{"/system be brief", false, func(s *session) { s.system, s.history = "be brief", nil }},
		{"/temp 0.2", false, func(s *session) { s.sp.Temperature = 0.2 }},
		{"/temp warm", false, none},
		{"/topk 5", false, func(s *session) { s.sp.TopK = 5 }},
		{"/topk many", false, none},
		{"/topp 0.5", false, func(s *session) { s.sp.TopP = 0.5 }},
		{"/seed 42", false, func(s *session) { s.sp.Seed = 42 }},
		{"/max 7", false, func(s *session) { s.maxTok = 7 }},
		{"/max", false, none},
		{"/think off", false, func(s *session) { s.think = chat.ThinkOff }},
		{"/think sideways", false, none},
		{"/effort high", false, func(s *session) { s.effort = "high" }},
		{"/json", false, func(s *session) { s.jsonOut = true }},
		{"/help", false, none},
		{"/params", false, none},
		{"/frobnicate", false, none},
		{"/demo no-such-demo", false, none},
	} {
		s, want := base(), base()
		c.change(want)
		var quits bool
		quietly(t, func() { quits = s.command(c.line) })
		if quits != c.quits || !reflect.DeepEqual(s, want) {
			t.Errorf("%q: quits %v (want %v)\n session %+v\n want    %+v", c.line, quits, c.quits, *s, *want)
		}
	}
	s := base()
	quietly(t, func() { s.command("/json"); s.command("/json") })
	if s.jsonOut {
		t.Error("/json twice left JSON output on: it toggles")
	}
}

// /demo runs its canned prompt as one turn against the model, the prompt and the reply joining the history. A JSON demo's turn
// is JSON-constrained (its reply opens a JSON value, which this model's unconstrained replies do not: the plain demo's reply is
// held to that too), and the JSON setting is back off after it, since the demo's setting is for its own turn.
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
		t.Fatalf("history after /demo: %+v, want the demo's prompt and a reply", s.history)
	}
	if reply := s.history[1].content; !opensJSONValue(reply) {
		t.Errorf("the JSON demo's reply %q does not open a JSON value: the turn was not JSON-constrained", reply)
	}
	if s.jsonOut {
		t.Error("the JSON demo left JSON output on")
	}
	s.history = nil
	quietly(t, func() { s.out = os.Stdout; s.command("/demo " + plainDemo(t)) })
	if len(s.history) != 2 || opensJSONValue(s.history[1].content) {
		t.Errorf("a demo without JSON replied %+v, as a constrained turn would: the check above cannot tell the two apart", s.history)
	}
}

// plainDemo is a demo with neither a system prompt nor JSON output.
func plainDemo(t *testing.T) string {
	t.Helper()
	for _, d := range demos {
		if !d.json && d.system == "" {
			return d.name
		}
	}
	t.Fatal("no plain demo")
	return ""
}

// opensJSONValue reports whether s, after leading space, starts the way a JSON value does.
func opensJSONValue(s string) bool {
	s = strings.TrimLeft(s, " \t\r\n")
	return s != "" && strings.ContainsRune(`{["-0123456789tfn`, rune(s[0]))
}

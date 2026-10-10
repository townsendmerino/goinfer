package chatapp

import (
	"bytes"
	"flag"
	"strings"
	"testing"
)

// R26 (docs/tasks/task-first-hour.md): through the REPL loop itself (the caller of every print), a scripted session
// (piped stdin) must print the answer and nothing else: no banner, `you>` label, ANSI escapes or trailing `bye`. An
// interactive one is unchanged. Origin: docs/code-notes/internal-chatapp.md#TestRepl_plainWhenScripted.
func TestRepl_plainWhenScripted(t *testing.T) {
	run := func(plain bool) string {
		var out bytes.Buffer
		s := &session{plain: plain, out: &out, system: "sys"}
		s.genFn = func() string {
			(&replyPrinter{s: s}).chunk("the answer") // what generate prints, through the same printer
			return "the answer"
		}
		s.repl(strings.NewReader("hello\n/quit\n"))
		return out.String()
	}

	got := run(true)
	if got != "the answer" {
		t.Errorf("plain REPL output = %q, want exactly the answer", got)
	}
	for _, bad := range []string{"\x1b", "you>", "bye", "goinfer chat", "system:"} {
		if strings.Contains(got, bad) {
			t.Errorf("plain output contains %q: %q", bad, got)
		}
	}

	// The control: the interactive REPL still has all of it, so the assertions above are able to fail.
	tty := run(false)
	for _, want := range []string{"\x1b[1myou>", "goinfer chat", "system: sys", "bye", "\x1b[36m"} {
		if !strings.Contains(tty, want) {
			t.Errorf("the interactive REPL lost %q: %q", want, tty)
		}
	}

	// EOF with no /quit: plain says nothing, interactive says bye.
	for _, plain := range []bool{true, false} {
		var out bytes.Buffer
		s := &session{plain: plain, out: &out}
		s.repl(strings.NewReader(""))
		if has := strings.Contains(out.String(), "bye"); has == plain {
			t.Errorf("plain=%v: EOF output %q (bye present = %v)", plain, out.String(), has)
		}
	}
}

// The reply printer: plain keeps stdout to the answer; the reasoning goes to the other writer, uncoloured. Interactive colours both on stdout.
func TestReplyPrinter_plainSeparatesReasoningFromTheAnswer(t *testing.T) {
	feed := func(s *session) {
		p := &replyPrinter{s: s}
		p.reasoning("let me think")
		p.reasoning(" more")
		p.chunk("42")
		p.chunk(".")
		p.end()
	}

	var out, errw bytes.Buffer
	feed(&session{plain: true, showThinking: true, out: &out, errw: &errw})
	if out.String() != "42.\n" {
		t.Errorf("plain stdout = %q, want only the answer and a newline", out.String())
	}
	if !strings.Contains(errw.String(), "let me think more") || strings.Contains(errw.String(), "\x1b") {
		t.Errorf("plain reasoning stream = %q, want the reasoning without escapes", errw.String())
	}

	out.Reset()
	errw.Reset()
	feed(&session{plain: true, showThinking: false, out: &out, errw: &errw})
	if out.String() != "42.\n" || !strings.Contains(errw.String(), "(thinking…)") || strings.Contains(errw.String(), "let me think") {
		t.Errorf("hidden reasoning: stdout %q, marker stream %q", out.String(), errw.String())
	}

	out.Reset()
	feed(&session{plain: false, showThinking: true, out: &out})
	for _, want := range []string{"\x1b[2mlet me think more", "\x1b[0m\n\n", "\x1b[36m42.", "\x1b[0m\n"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("interactive output lost %q: %q", want, out.String())
		}
	}
}

// -p: one prompt in, the answer out, exit 0; no answer is exit 1.
func TestOneShot(t *testing.T) {
	var out bytes.Buffer
	s := &session{plain: true, out: &out}
	s.genFn = func() string {
		if len(s.history) != 1 || s.history[0].role != "user" || s.history[0].content != "Say hi." {
			t.Errorf("history = %+v, want the one user prompt", s.history)
		}
		return "hi"
	}
	if code := s.oneShot("Say hi."); code != 0 {
		t.Errorf("exit code %d for an answered prompt", code)
	}
	s2 := &session{plain: true, out: &out, genFn: func() string { return "  " }}
	if code := s2.oneShot("x"); code != 1 {
		t.Errorf("exit code %d for an empty answer, want 1", code)
	}
}

// -p is on chat's command line (a test passes a fresh FlagSet, so this is the real list).
func TestPromptFlagRegistered(t *testing.T) {
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	registerFlags(fs)
	if f := fs.Lookup("p"); f == nil || f.DefValue != "" {
		t.Errorf("-p flag = %+v", f)
	}
}

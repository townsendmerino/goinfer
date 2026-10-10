package chatapp

import (
	"flag"
	"testing"
)

// chatFlagSet is chat's real command line on a fresh flag set (registerFlags), so these tests read the
// same list serveOnlyInvocation does at run time.
func chatFlagSet() *flag.FlagSet {
	fs := flag.NewFlagSet("goinfer-chat", flag.ContinueOnError)
	registerFlags(fs)
	return fs
}

// TestServeOnlyInvocation pins that a serve-only subcommand or flag typed at chat (`serve`, `-web`) is named, so the
// caller can say it belongs to the other binary, instead of an ignored subcommand or a bare "flag provided but not
// defined"; a legitimate chat invocation, including every flag the two binaries share, is NEVER redirected. Origin:
// docs/code-notes/internal-chatapp.md#TestServeOnlyInvocation.
func TestServeOnlyInvocation(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{"the subcommand the tester typed", []string{"serve"}, "serve"},
		{"the flag the README's example used", []string{"-web"}, "-web"},
		{"double dash", []string{"--web"}, "--web"},
		{"flag with an = value", []string{"--addr=0.0.0.0:8080"}, "--addr"},
		{"serve-only flag after a shared one", []string{"--model", "m.gguf", "--api-key", "x"}, "--api-key"},
		// --stream-weights is shared with serve through internal/loadflags, so it must fall through to chat.
		{"a model-loading flag chat now shares", []string{"-stream-weights"}, ""},
		{"and one that takes a value, with its value", []string{"--moe-cache-slots", "8", "--ctx", "16384"}, ""},

		// The other direction matters more: a legitimate chat invocation must NEVER be redirected.
		// Every flag both binaries share has to fall through, or this "help" breaks the tool.
		{"plain chat", []string{"--model", "m.gguf"}, ""},
		{"shared flags only", []string{"--model", "m.gguf", "--backend", "cuda", "--quant", "int4"}, ""},
		{"chat's own pull", []string{"pull", "hf:owner/repo:q4_k_m"}, ""},
		{"no args", nil, ""},
		{"a path that merely contains the word serve", []string{"--model", "/models/serve/m.gguf"}, ""},
		// N-77: a token that is a flag's VALUE, not a flag or subcommand, must not trigger the redirect: a chat flag's value
		// equal to a serveOnly word (most plausibly "serve" itself, via --system).
		{"a flag VALUE that happens to equal a serveOnly word", []string{"--system", "serve"}, ""},
		{"same, with a serve-only flag genuinely after it", []string{"--system", "serve", "--web"}, "--web"},
		{"a genuine serve-only flag is still caught when NOT preceded by a value-taking flag",
			[]string{"--model", "m.gguf", "serve"}, "serve"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := serveOnlyInvocation(tc.args, chatFlagSet()); got != tc.want {
				t.Errorf("serveOnlyInvocation(%q) = %q, want %q", tc.args, got, tc.want)
			}
		})
	}
}

// The redirect list must not overlap with flags chat actually has, or a working invocation gets
// turned away. Checked against chat's REGISTERED flags — every one, in both spellings — rather than a
// list typed out here, because the failure is silent: the binary simply stops accepting something it
// used to, and a typed-out list goes stale the day a flag is added.
func TestServeOnlyInvocation_neverClaimsAChatFlag(t *testing.T) {
	fs := chatFlagSet()
	n := 0
	fs.VisitAll(func(f *flag.Flag) {
		for _, spelling := range []string{"-" + f.Name, "--" + f.Name} {
			n++
			if got := serveOnlyInvocation([]string{spelling, "x"}, fs); got != "" {
				t.Errorf("%s is a chat flag but was redirected to serve as %q", spelling, got)
			}
		}
	})
	if n < 40 {
		t.Fatalf("only %d flag spellings checked — registerFlags registered fewer flags than chat has", n)
	}
}

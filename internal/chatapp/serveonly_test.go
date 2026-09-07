package chatapp

import "testing"

// Cold-user run 2026-09-06, scenario B, 06:37:23 — the tester's FIRST error of that leg, and the
// reason it started badly. `goinfer-chat serve` ignored the subcommand and complained about
// --model; `goinfer-chat -web` said "flag provided but not defined: -web". Neither said the two
// belong to a different binary, and the README's own examples used them, so the tester concluded
// the released binary was broken and went hunting through pkg.go.dev for a server.
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
		{"the flag whose absence cost the run its worst dead end", []string{"-stream-weights"}, "-stream-weights"},

		// The other direction matters more: a legitimate chat invocation must NEVER be redirected.
		// Every flag both binaries share has to fall through, or this "help" breaks the tool.
		{"plain chat", []string{"--model", "m.gguf"}, ""},
		{"shared flags only", []string{"--model", "m.gguf", "--backend", "cuda", "--quant", "int4"}, ""},
		{"chat's own pull", []string{"pull", "hf:owner/repo:q4_k_m"}, ""},
		{"no args", nil, ""},
		{"a path that merely contains the word serve", []string{"--model", "/models/serve/m.gguf"}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := serveOnlyInvocation(tc.args); got != tc.want {
				t.Errorf("serveOnlyInvocation(%q) = %q, want %q", tc.args, got, tc.want)
			}
		})
	}
}

// The redirect list must not overlap with flags chat actually has, or a working invocation gets
// turned away. Checked against the list rather than by eye, because the failure is silent: the
// binary simply stops accepting something it used to.
func TestServeOnlyInvocation_neverClaimsAChatFlag(t *testing.T) {
	// Flags chat defines itself; every one must fall through untouched.
	for _, f := range []string{
		"--model", "-model", "--backend", "-backend", "--quant", "-quant", "--lora", "-lora",
		"--system", "-system", "--max", "-max", "--temp", "-temp", "--top-k", "--top-p", "--seed",
	} {
		if got := serveOnlyInvocation([]string{f, "x"}); got != "" {
			t.Errorf("%s is a chat flag but was redirected to serve as %q", f, got)
		}
	}
}

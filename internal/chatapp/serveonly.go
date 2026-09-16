package chatapp

import "strings"

// serveOnlyInvocation reports the first argument that belongs to goinfer-serve rather than to the
// chat runtime, or "" if none does. It is deliberately a SMALL list — the flags and subcommands a
// person is most likely to carry over from a README, a blog post, or their own shell history.
//
// It matches only what serve has and chat does not. A flag both binaries share (--model, --backend,
// --quant) must never appear here, or a legitimate chat invocation gets redirected.
func serveOnlyInvocation(args []string) string {
	serveOnly := map[string]bool{
		// The subcommand the cold-user tester actually typed.
		"serve": true,
		// serve-only flags, in both spellings Go's flag package accepts.
		"-web": true, "--web": true,
		"-addr": true, "--addr": true,
		"-api-key": true, "--api-key": true,
		"-stream-weights": true, "--stream-weights": true,
		"-allow-admin": true, "--allow-admin": true,
		"-embed-model": true, "--embed-model": true,
		"-require-backend": true, "--require-backend": true,
	}
	// N-77 (docs/audit-2026-09-10.md): this used to scan every argv token uniformly, including a
	// VALUE the user passed to a chat flag — `--system serve` set the system prompt to the
	// literal word "serve" and got redirected to goinfer-serve as if "serve" were a subcommand.
	// Chat's own value-taking flags (the space-separated `--flag value` form only — `--flag=value`
	// is already one token and never ambiguous) consume the next token; skip checking it here the
	// same way a real flag parser would consume it, rather than matching it against serveOnly.
	chatValueFlags := map[string]bool{
		"-model": true, "--model": true,
		"-system": true, "--system": true,
		"-backend": true, "--backend": true,
		"-quant": true, "--quant": true,
		"-lora": true, "--lora": true,
		"-max": true, "--max": true,
		"-temp": true, "--temp": true,
		"-top-k": true, "--top-k": true,
		"-top-p": true, "--top-p": true,
		"-min-p": true, "--min-p": true,
		"-repeat-penalty": true, "--repeat-penalty": true,
		"-presence-penalty": true, "--presence-penalty": true,
		"-frequency-penalty": true, "--frequency-penalty": true,
		"-repeat-last-n": true, "--repeat-last-n": true,
		"-schema": true, "--schema": true,
		"-draft": true, "--draft": true,
		"-spec-k": true, "--spec-k": true,
	}
	skipNext := false
	for _, a := range args {
		if skipNext {
			skipNext = false
			continue
		}
		// `--addr=x` and `--addr x` are the same flag to the user.
		name, _, hasEq := strings.Cut(a, "=")
		if serveOnly[name] {
			return name
		}
		if !hasEq && chatValueFlags[name] { // bare "--flag", no "=value": the NEXT token is its value
			skipNext = true
		}
	}
	return ""
}

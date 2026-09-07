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
	for _, a := range args {
		// `--addr=x` and `--addr x` are the same flag to the user.
		name, _, _ := strings.Cut(a, "=")
		if serveOnly[name] {
			return name
		}
	}
	return ""
}

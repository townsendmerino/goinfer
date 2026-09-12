package serveapp

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"time"
)

// isAdminCLICmd reports whether args[0] (when present) names one of K5's one-word admin
// subcommands (docs/task-halt-2026-09.md), dispatched from Main() the same way `pull`/`check`
// already are — before flag.Parse, since each gets its own flag set.
func isAdminCLICmd(cmd string) bool {
	switch cmd {
	case "status", "ls", "cancel", "halt", "resume":
		return true
	}
	return false
}

// runAdminCLI implements `<binary> status|ls|cancel <id> [reason]|halt [reason]|resume`, talking
// to the admin socket a running `serve -admin-socket ...` set up (K5). No -api-key: the socket's
// file permissions are the auth, so this CLI needs none either — it just has to be running as
// whichever user can open the socket file.
func runAdminCLI(cmd string, args []string, self string) int {
	fs := flag.NewFlagSet(self+" "+cmd, flag.ExitOnError)
	// NOTE: -admin-socket must come BEFORE any positional reason text (`halt -admin-socket
	// <path> "reason"`, not the reverse) — Go's flag package stops parsing flags at the first
	// non-flag argument, so a reason typed first silently falls back to the default path
	// instead of erroring. Documented in the flag's own --help text too.
	sockPath := fs.String("admin-socket", defaultAdminSocketPath(),
		"path to the running server's admin socket; must match its own -admin-socket. Must be given before any positional argument (e.g. a halt reason)")
	fs.Parse(args)
	rest := fs.Args()

	client := &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
				var d net.Dialer
				return d.DialContext(ctx, "unix", *sockPath)
			},
		},
		// Longer than halt's own 30s bounded wait for quiescence (halt.go) so a real halt under
		// load doesn't time out the CLI call that's waiting to report it.
		Timeout: 35 * time.Second,
	}
	// The host in this URL is never resolved (DialContext above always dials the Unix socket
	// instead) — it exists only because http.NewRequest requires a syntactically valid URL.
	const base = "http://admin-socket"

	var method, path, body string
	switch cmd {
	case "status":
		method, path = "GET", "/admin/status"
	case "ls":
		method, path = "GET", "/admin/generations"
	case "cancel":
		if len(rest) < 1 {
			fmt.Fprintf(os.Stderr, "usage: %s cancel <id> [reason]\n", self)
			return 2
		}
		reason := "operator cancel"
		if len(rest) > 1 {
			reason = strings.Join(rest[1:], " ")
		}
		method, path = "POST", "/admin/generations/"+rest[0]+"/cancel"
		b, _ := json.Marshal(map[string]string{"reason": reason})
		body = string(b)
	case "halt":
		reason := "operator halt"
		if len(rest) > 0 {
			reason = strings.Join(rest, " ")
		}
		method, path = "POST", "/admin/halt"
		b, _ := json.Marshal(map[string]string{"reason": reason})
		body = string(b)
	case "resume":
		method, path = "POST", "/admin/resume"
	default:
		fmt.Fprintf(os.Stderr, "unknown admin command %q\n", cmd)
		return 2
	}

	var bodyReader io.Reader
	if body != "" {
		bodyReader = strings.NewReader(body)
	}
	req, err := http.NewRequest(method, base+path, bodyReader)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "connecting to admin socket %s: %v (is serve running with -admin-socket %s?)\n", *sockPath, err, *sockPath)
		return 1
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	// Best-effort pretty-print for a human at a terminal; raw bytes on anything that doesn't
	// decode as JSON (shouldn't happen against this server, but this is output formatting, not a
	// correctness path, so falling back silently is fine).
	var parsed map[string]any
	if json.Unmarshal(out, &parsed) == nil {
		if pretty, err := json.MarshalIndent(parsed, "", "  "); err == nil {
			out = pretty
		}
	}
	fmt.Println(string(out))
	if resp.StatusCode >= 400 {
		return 1
	}
	return 0
}

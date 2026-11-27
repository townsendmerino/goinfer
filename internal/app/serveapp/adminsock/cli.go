package adminsock

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

// IsCLICmd reports whether cmd names one of the admin socket subcommands.
func IsCLICmd(cmd string) bool {
	switch cmd {
	case "status", "ls", "cancel", "halt", "resume":
		return true
	}
	return false
}

// NewUnixHTTPClient returns an http.Client that dials only the Unix socket at path.
func NewUnixHTTPClient(path string, timeout time.Duration) *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
				var d net.Dialer
				return d.DialContext(ctx, "unix", path)
			},
		},
		Timeout: timeout,
	}
}

// RunCLI executes `status|ls|cancel <id> [reason]|halt [reason]|resume` over the
// admin Unix socket and returns the process exit code.
func RunCLI(cmd string, args []string, self string) int {
	fs := flag.NewFlagSet(self+" "+cmd, flag.ExitOnError)
	sockPath := fs.String("admin-socket", DefaultSocketPath(),
		"path to the running server's admin socket; must match its own -admin-socket. Must be given before any positional argument (e.g. a halt reason)")
	_ = fs.Parse(args)
	rest := fs.Args()

	client := NewUnixHTTPClient(*sockPath, 35*time.Second)
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

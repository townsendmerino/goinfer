// Command claude-guard is a Claude Code PreToolUse hook that refuses the commands these repos' rules forbid and names
// the right one. Claude Code passes the tool call as JSON on stdin; a refusal exits 2 with its message on stderr, which
// Claude Code shows the session in place of running the tool. Anything else, including input it cannot parse, exits 0.
//
// Each refusal is appended to the log (-log, default ~/.local/state/claude-guard/refusals.jsonl) as one JSON line: time,
// rule id, tool, cwd, session and command. The monthly rule review reads it to find guards that no longer fire.
//
//	go build -o ~/go/bin/claude-guard ./cmd/claude-guard
//
// The rules and their ids are in rules.go; docs/tasks/task-claude-code-hygiene-2026-10.md (DX4) has the wiring.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// hookInput is the part of Claude Code's PreToolUse payload the rules read.
type hookInput struct {
	SessionID string `json:"session_id"`
	Cwd       string `json:"cwd"`
	ToolName  string `json:"tool_name"`
	ToolInput struct {
		Command string `json:"command"`
	} `json:"tool_input"`
}

// logEntry is one line of the refusal log.
type logEntry struct {
	Time    string `json:"time"`
	Rule    string `json:"rule"`
	Tool    string `json:"tool"`
	Cwd     string `json:"cwd"`
	Session string `json:"session"`
	Command string `json:"command,omitempty"`
}

func main() {
	logPath := flag.String("log", defaultLog(), "refusal log (JSON lines); empty disables logging")
	flag.Parse()
	os.Exit(run(os.Stdin, os.Stderr, *logPath, time.Now()))
}

func defaultLog() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".local", "state", "claude-guard", "refusals.jsonl")
}

// run reads one hook payload and returns the exit code: 2 to refuse, 0 to allow. A log that cannot be written does not
// change the decision; it is reported on stderr with the refusal.
func run(stdin io.Reader, stderr io.Writer, logPath string, now time.Time) int {
	var in hookInput
	if err := json.NewDecoder(stdin).Decode(&in); err != nil {
		return 0
	}
	r := check(in.ToolName, in.ToolInput.Command)
	if r == nil {
		return 0
	}
	fmt.Fprintln(stderr, "claude-guard refused this call. "+r.Message)
	if logPath != "" {
		if err := appendLog(logPath, logEntry{
			Time: now.UTC().Format(time.RFC3339), Rule: r.Rule, Tool: in.ToolName, Cwd: in.Cwd,
			Session: in.SessionID, Command: truncate(in.ToolInput.Command, 2000),
		}); err != nil {
			fmt.Fprintf(stderr, "(claude-guard could not write its log %s: %v)\n", logPath, err)
		}
	}
	return 2
}

func appendLog(path string, e logEntry) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false) // a command's && stays readable
	if err := enc.Encode(e); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(b.Bytes()); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

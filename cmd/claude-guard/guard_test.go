package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Each rule's refused forms, and the near misses it must allow: a quoted mention, a heredoc body, a neighboring flag.
var cases = []struct {
	rule    string // "" = allowed
	command string
}{
	{"GA1", "git add -A"},
	{"GA1", "git add --all"},
	{"GA1", "git add ."},
	{"GA1", "git add -vA"},
	{"GA1", "git -C /repo add -A"},
	{"GA1", "cd ~/dx01-wt/x && git add . && git commit -m wip"},
	{"GA1", "git add -- ."},
	{"GA1", "ssh nobara 'cd ~/goinfer && git add -A'"},
	{"GA1", "bash -lc 'git add -A'"},
	{"", "git add cmd/lint/lint.go scripts/touched.py"},
	{"", "git add ./cmd/lint"},
	{"", `git commit -m "never git add -A"`},
	{"", "echo 'git add .'"},
	{"", "git add -p cmd/x.go"},
	{"", "git commit -q -F - <<'EOF'\nDX4: refuses git add -A and --no-verify\n\ngit add . is GA1\nEOF"},
	{"", "git commit -m \"$(cat <<'EOF'\ngit add -A --no-verify\nEOF\n)\""},

	{"NV1", "git push --no-verify origin x"},
	{"NV1", "git commit --no-verify -m x"},
	{"NV1", "git commit -n -m x"},
	{"NV1", "git commit -anm x"},
	{"NV1", "git -c core.hooksPath=/dev/null push"},
	{"NV1", "git rebase --continue --no-verify"},
	{"NV1", "ssh nobara 'cd ~/goinfer && git push --no-verify'"},
	{"", "git push -n origin x"},
	{"", "git commit -m 'skip --no-verify forever'"},
	{"", "git commit -mn"},
	{"", "git log --grep=--no-verify"},
	{"", "git config core.hooksPath"},

	{"SC1", "GOOS=linux go run honnef.co/go/tools/cmd/staticcheck@v0.8.1 ./..."},
	{"SC1", "GOOS=linux GOARCH=amd64 CGO_ENABLED=0 ~/go/bin/staticcheck -tags cuda ./..."},
	{"SC1", "env GOOS=linux staticcheck ./cuda/..."},
	{"SC1", "export GOOS=linux; staticcheck ./..."},
	{"SC1", "python3 scripts/timing_lock.py run --shared --label x -- env GOOS=linux staticcheck ./..."},
	{"", "staticcheck ./..."},
	{"", "GOOS=linux go vet -tags cuda ./cuda/..."},
	{"", "go run ./cmd/lint"},
	{"", "GOOS=linux go build ./... && staticcheck ./..."},
	{"", `echo "GOOS=linux staticcheck"`},

	{"TL1", "rm ~/.goinfer-timing.lock"},
	{"TL1", `rm -f "$HOME/.goinfer-timing.lock"`},
	{"TL1", "mv ~/.goinfer-timing.lock /tmp/x"},
	{"TL1", "ssh nobara 'rm -f ~/.goinfer-timing.lock'"},
	{"", "python3 scripts/timing_lock.py status"},
	{"", "cat ~/.goinfer-timing.lock"},
	{"", "ls -la ~/.goinfer-timing.lock && rm -f /tmp/other.lock"},

	{"", "git status"},
	{"", "echo 'unterminated"},
}

func TestRules(t *testing.T) {
	for _, c := range cases {
		r := check("Bash", c.command)
		got := ""
		if r != nil {
			got = r.Rule
		}
		if got != c.rule {
			t.Errorf("%q: refused by %q, want %q", c.command, got, c.rule)
		}
		if r != nil && !strings.HasPrefix(r.Message, r.Rule+": ") {
			t.Errorf("%s's message does not start with its id: %q", r.Rule, r.Message)
		}
	}
}

// TestEveryRuleHasACase holds the table to the rule set: a rule with no refused case is untested, and a rule that is
// only ever allowed would read as dead in the monthly review.
func TestEveryRuleHasACase(t *testing.T) {
	seen := map[string]bool{}
	for _, c := range cases {
		seen[c.rule] = true
	}
	for _, id := range []string{"GA1", "NV1", "SC1", "TL1"} {
		if !seen[id] {
			t.Errorf("rule %s has no refused case", id)
		}
	}
}

func TestEnterWorktreeRefused(t *testing.T) {
	if r := check("EnterWorktree", ""); r == nil || r.Rule != "WT1" {
		t.Errorf("EnterWorktree: %v, want WT1", r)
	}
	if r := check("Edit", "git add -A"); r != nil {
		t.Errorf("a non-Bash tool's input was parsed as a command: %v", r)
	}
}

func payload(t *testing.T, tool, command string) *bytes.Reader {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"session_id": "s1", "cwd": "/repo", "hook_event_name": "PreToolUse",
		"tool_name": tool, "tool_input": map[string]string{"command": command},
	})
	if err != nil {
		t.Fatal(err)
	}
	return bytes.NewReader(b)
}

// TestRun_refusalExitsTwoAndLogs drives the hook as Claude Code does: a refusal exits 2, names the rule on stderr and
// appends one log line with the rule id; an allowed call exits 0 and logs nothing.
func TestRun_refusalExitsTwoAndLogs(t *testing.T) {
	log := filepath.Join(t.TempDir(), "state", "refusals.jsonl")
	now := time.Date(2026, 10, 30, 17, 0, 0, 0, time.UTC)
	var stderr bytes.Buffer
	if code := run(payload(t, "Bash", "git status"), &stderr, log, now); code != 0 || stderr.Len() != 0 {
		t.Fatalf("allowed call: exit %d, stderr %q", code, stderr.String())
	}
	if _, err := os.Stat(log); !os.IsNotExist(err) {
		t.Fatalf("an allowed call wrote the log: %v", err)
	}
	if code := run(payload(t, "Bash", "git add -A && git push"), &stderr, log, now); code != 2 {
		t.Fatalf("refused call: exit %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "GA1:") || !strings.Contains(stderr.String(), "git add <path>") {
		t.Errorf("stderr does not name the rule and the right command: %q", stderr.String())
	}
	if code := run(payload(t, "EnterWorktree", ""), &stderr, log, now); code != 2 {
		t.Fatalf("EnterWorktree: exit %d, want 2", code)
	}
	b, err := os.ReadFile(log)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) != 2 {
		t.Fatalf("log has %d lines, want 2:\n%s", len(lines), b)
	}
	if !strings.Contains(lines[0], "&&") {
		t.Errorf("the log escaped the command: %s", lines[0])
	}
	var e logEntry
	if err := json.Unmarshal([]byte(lines[0]), &e); err != nil {
		t.Fatal(err)
	}
	want := logEntry{Time: "2026-10-30T17:00:00Z", Rule: "GA1", Tool: "Bash", Cwd: "/repo", Session: "s1", Command: "git add -A && git push"}
	if e != want {
		t.Errorf("log line = %+v, want %+v", e, want)
	}
}

func TestRun_badInputAllows(t *testing.T) {
	var stderr bytes.Buffer
	if code := run(strings.NewReader("not json"), &stderr, "", time.Now()); code != 0 {
		t.Errorf("unparseable input: exit %d, want 0 (fail open)", code)
	}
}

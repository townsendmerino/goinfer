package chatapp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// mainArgsEnv, when set, makes the test binary run Main with these arguments (one per line) instead of the tests, so a test can
// drive the goinfer-chat command line in a child process: Main reads os.Args and may call os.Exit.
const mainArgsEnv = "CHATAPP_TEST_MAIN_ARGS"

func TestMain(m *testing.M) {
	if args, ok := os.LookupEnv(mainArgsEnv); ok {
		os.Args = append([]string{"goinfer-chat"}, strings.Split(args, "\n")...)
		Main()
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// runMain runs goinfer-chat with args and stdin in a child process, and returns its exit code and output. A child gets a
// minute.
func runMain(t *testing.T, stdin string, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0])
	cmd.Env = append(os.Environ(), mainArgsEnv+"="+strings.Join(args, "\n"))
	cmd.Stdin = strings.NewReader(stdin)
	var o, e bytes.Buffer
	cmd.Stdout, cmd.Stderr = &o, &e
	err := cmd.Run()
	var exit *exec.ExitError
	switch {
	case err == nil:
	case errors.As(err, &exit):
		code = exit.ExitCode()
	default:
		t.Fatalf("running goinfer-chat %q: %v\nstderr:\n%s", args, err, e.String())
	}
	if ctx.Err() != nil {
		t.Fatalf("goinfer-chat %q did not finish in a minute\nstderr:\n%s", args, e.String())
	}
	return code, o.String(), e.String()
}

// -p prints one answer on stdout and exits 0. stdout is the answer and nothing else (the load lines are stderr), and a greedy
// answer is the same on a second run.
func TestMain_oneShotPromptAnswersOnStdoutAlone(t *testing.T) {
	dir := tinyChatModel(t)
	args := []string{"--model", dir, "--no-selftest", "-p", "hello", "--max", "6", "--temp", "0"}
	code, first, stderr := runMain(t, "", args...)
	if code != 0 || strings.TrimSpace(first) == "" {
		t.Fatalf("exit %d, stdout %q\nstderr:\n%s", code, first, stderr)
	}
	for _, banner := range []string{"loaded ", "chat template", "you>"} {
		if strings.Contains(first, banner) {
			t.Errorf("stdout holds %q: it must be the answer alone\n%s", banner, first)
		}
	}
	if code, second, _ := runMain(t, "", args...); code != 0 || second != first {
		t.Errorf("a second greedy run: exit %d, %q; the first said %q", code, second, first)
	}
}

// The REPL reads commands and prompts from stdin: /max sets the reply length the next prompt gets, and /quit ends it, exit 0.
func TestMain_replTakesCommandsAndPromptsFromStdin(t *testing.T) {
	dir := tinyChatModel(t)
	code, stdout, stderr := runMain(t, "/temp 0\n/max 3\nhello\n/quit\n", "--model", dir, "--no-selftest")
	if code != 0 || strings.TrimSpace(stdout) == "" {
		t.Fatalf("exit %d, stdout %q\nstderr:\n%s", code, stdout, stderr)
	}
	if !strings.Contains(stderr, "[3 tok") {
		t.Errorf("the reply after /max 3 was not 3 tokens\nstderr:\n%s", stderr)
	}
}

// --batch writes one response per line that ran to -o and each refused line to the errors file, exit 1 while any line fails. A
// rerun skips the finished lines, so -o gains nothing twice.
func TestMain_batchWritesEachLineAndResumes(t *testing.T) {
	dir := tinyChatModel(t)
	work := t.TempDir()
	in, out := filepath.Join(work, "in.jsonl"), filepath.Join(work, "out.jsonl")
	line := func(id, extra string) string {
		return `{"custom_id":"` + id + `","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0` + extra + `}}`
	}
	if err := os.WriteFile(in, []byte(line("a", "")+"\n"+line("b", "")+"\n"+line("c", `,"n":2`)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	args := []string{"--model", dir, "--no-selftest", "--batch", in, "-o", out}
	code, _, stderr := runMain(t, "", args...)
	if code != 1 {
		t.Fatalf("exit %d with a refused line, want 1\nstderr:\n%s", code, stderr)
	}
	// results is each line's custom_id and what it holds: "200" and the reply for a response, the message for an error.
	results := func(path string) map[string]string {
		t.Helper()
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		got := map[string]string{}
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			var l struct {
				CustomID string `json:"custom_id"`
				Response *struct {
					StatusCode int `json:"status_code"`
					Body       struct {
						Choices []struct {
							Message struct {
								Content string `json:"content"`
							} `json:"message"`
						} `json:"choices"`
					} `json:"body"`
				} `json:"response"`
				Error *struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal(sc.Bytes(), &l); err != nil {
				t.Fatalf("%s: %v in %q", path, err, sc.Text())
			}
			switch {
			case l.Response != nil && len(l.Response.Body.Choices) == 1:
				got[l.CustomID] = fmt.Sprintf("%d %q", l.Response.StatusCode, l.Response.Body.Choices[0].Message.Content)
			case l.Error != nil:
				got[l.CustomID] = l.Error.Message
			default:
				got[l.CustomID] = "neither a response nor an error: " + sc.Text()
			}
		}
		return got
	}
	got := results(out)
	if len(got) != 2 || !strings.HasPrefix(got["a"], "200 ") || got["a"] != got["b"] {
		t.Errorf("-o holds %v, want a and b at 200 with the same greedy reply", got)
	}
	if got := results(strings.TrimSuffix(out, ".jsonl") + ".errors.jsonl"); len(got) != 1 || !strings.Contains(got["c"], "n must be 1") {
		t.Errorf("the errors file holds %v, want c refused for n", got)
	}
	before, _ := os.ReadFile(out)
	code, _, stderr = runMain(t, "", args...)
	after, _ := os.ReadFile(out)
	if code != 1 || !bytes.Equal(before, after) || !strings.Contains(stderr, "2 already done") {
		t.Errorf("rerun: exit %d, -o changed %v\nstderr:\n%s", code, !bytes.Equal(before, after), stderr)
	}
}

// A model path that does not exist fails at load, naming it, with a nonzero exit and nothing on stdout.
func TestMain_aMissingModelFailsNamingIt(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "no-such-model")
	code, stdout, stderr := runMain(t, "", "--model", missing, "--no-selftest", "-p", "hi")
	if code == 0 || !strings.Contains(stderr, "no-such-model") || stdout != "" {
		t.Errorf("exit %d, stdout %q\nstderr:\n%s", code, stdout, stderr)
	}
}

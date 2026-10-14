package connect

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// The cap the server enforces, as /v1/models publishes it, is what -write gives Claude Code as its window, and the
// output cap it writes is claudeCodeOutput at every window: it does not scale with the window.
func TestRun_claudeCodeWindowFromServedCap(t *testing.T) {
	for _, window := range []int{28672, 40960, 131072} {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte(`{"data":[{"id":"coder","context_window":` + strconv.Itoa(window) + `}]}`))
		}))
		path := filepath.Join(t.TempDir(), "settings.json")
		var out bytes.Buffer
		if rc := Run([]string{"claude-code", "-file", path, "-write", ts.URL}, "goinfer-serve", &out); rc != 0 {
			t.Fatalf("window %d: rc %d\n%s", window, rc, out.String())
		}
		ts.Close()
		raw, _ := os.ReadFile(path)
		var got struct {
			Env map[string]string `json:"env"`
		}
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		if got.Env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] != strconv.Itoa(window) || got.Env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] != "4096" {
			t.Errorf("window %d: wrote %v, want the window and an output cap of 4096", window, got.Env)
		}
		if !strings.Contains(out.String(), "(output limit 4096)") {
			t.Errorf("window %d: the printout does not give the cap it writes:\n%s", window, out.String())
		}
	}
}

// The threshold is the first request, the written output cap and the margin, and none of them moves with the window,
// so a quarter-of-the-window reserve cannot come back unnoticed.
func TestClaudeCodeThreshold(t *testing.T) {
	h := harnesses["claude-code"]
	if claudeCodeOutput != 4096 {
		t.Errorf("claudeCodeOutput = %d, want 4096", claudeCodeOutput)
	}
	if got, want := h.claudeCodeThreshold(), h.FirstRequest+claudeCodeOutput+h.FirstRequest/4; got != want || got != 29344 {
		t.Errorf("threshold = %d, want first request %d + output cap %d + margin %d = %d (29344)", got, h.FirstRequest, claudeCodeOutput, h.FirstRequest/4, want)
	}
	for _, window := range []int{8192, 32768, 131072, 1 << 20} {
		s := claudeCodeSettings("claude-code", settings{Window: window, Output: OutputLimit(window)})
		if s.Output != claudeCodeOutput {
			t.Errorf("window %d: output %d, want %d whatever the window", window, s.Output, claudeCodeOutput)
		}
	}
	if s := claudeCodeSettings("codex", settings{Window: 32768, Output: 8192}); s.Output != 8192 {
		t.Errorf("codex's output limit was changed to %d", s.Output)
	}
}

// Below the first request plus the output cap, a warning that names the -ctx; between that and the threshold, a note;
// above, nothing. A model whose own maximum cannot reach the first request plus the cap is told so instead of a -ctx.
func TestClaudeCodeWindowLevels(t *testing.T) {
	level := func(window, max int) (string, string) {
		var out bytes.Buffer
		s := testSettings
		s.Window, s.Output, s.MaxPos = window, OutputLimit(window), max
		connect(&out, "claude-code", harnesses["claude-code"], filepath.Join(t.TempDir(), "settings.json"), s, false, false)
		switch o := out.String(); {
		case strings.Contains(o, "warning: Claude Code's first request"):
			return "warning", o
		case strings.Contains(o, "note: Claude Code"):
			return "note", o
		default:
			return "", o
		}
	}
	for _, c := range []struct {
		window, max int
		want        string
	}{
		{16384, 0, "warning"}, {24294, 0, "warning"}, {24295, 0, "note"}, {28672, 0, "note"}, {29343, 0, "note"},
		{29344, 0, ""}, {32768, 0, ""}, {33440, 0, ""}, {131072, 0, ""},
		{28672, 32768, "note"}, {32768, 32768, ""},
	} {
		if got, out := level(c.window, c.max); got != c.want {
			t.Errorf("window %d (model max %d): level %q, want %q\n%s", c.window, c.max, got, c.want, out)
		}
	}
	if _, out := level(16384, 0); !strings.Contains(out, "-ctx 29344 or more") {
		t.Errorf("the warning does not name -ctx 29344:\n%s", out)
	}
	warnLine := func(out string) string {
		for _, ln := range strings.Split(out, "\n") {
			if strings.Contains(ln, "warning: Claude Code's first request") {
				return ln
			}
		}
		return ""
	}
	if _, out := level(16384, 16384); !strings.Contains(warnLine(out), "cannot hold") || strings.Contains(warnLine(out), "-ctx") {
		t.Errorf("a model whose own maximum is 16384 was given a -ctx to pass:\n%s", out)
	}
	if _, out := level(16384, 26000); !strings.Contains(out, "-ctx 26000 or more") {
		t.Errorf("a model whose own maximum is 26000 was not pointed at it:\n%s", out)
	}
	if strings.Count(func() string { _, o := level(28672, 0); return o }(), "\n  note:") != 1 {
		t.Errorf("the note is not one line")
	}
	// The warnings are Claude Code's alone.
	var other bytes.Buffer
	s := testSettings
	s.Window, s.Output = 8192, OutputLimit(8192)
	connect(&other, "opencode", harnesses["opencode"], filepath.Join(t.TempDir(), "opencode.json"), s, false, false)
	if strings.Contains(other.String(), "Claude Code") {
		t.Errorf("opencode got Claude Code's warning:\n%s", other.String())
	}
}

// /v1/models' max_positions reaches the settings connect works from.
func TestProbe_maxPositions(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"data":[{"id":"coder","context_window":28672,"max_positions":40960}]}`))
	}))
	defer ts.Close()
	s, err := probe(ts.Client(), ts.URL, "", "")
	if err != nil || s.Window != 28672 || s.MaxPos != 40960 {
		t.Errorf("probe: %+v %v", s, err)
	}
}

// Claude Code ignores CLAUDE_CODE_MAX_CONTEXT_TOKENS for a model id it reads as a Claude model or as a [1m] variant, so
// connect says so for those ids and not for others.
func TestClaudeCodeIDWarning(t *testing.T) {
	for id, want := range map[string]bool{"coder": false, "qwen3-4b": false, "claude-sonnet-5-5": true, "Claude-local": true, "gw/claude-opus-4-8": true, "coder[1M]": true} {
		var out bytes.Buffer
		s := testSettings
		s.Model = id
		connect(&out, "claude-code", harnesses["claude-code"], filepath.Join(t.TempDir(), "settings.json"), s, false, false)
		if got := strings.Contains(out.String(), "may not apply"); got != want {
			t.Errorf("model %q: warned %v, want %v\n%s", id, got, want, out.String())
		}
	}
}

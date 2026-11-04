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
// output cap it writes is claudeCodeCap of that window.
func TestRun_claudeCodeWindowFromServedCap(t *testing.T) {
	for window, cap := range map[int]int{28672: 6425, 40960: 8192, 131072: 8192} {
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
		if got.Env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] != strconv.Itoa(window) || got.Env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] != strconv.Itoa(cap) {
			t.Errorf("window %d: wrote %v, want the window and an output cap of %d", window, got.Env, cap)
		}
		if !strings.Contains(out.String(), "(output limit "+strconv.Itoa(cap)+")") {
			t.Errorf("window %d: the printout does not give the cap it writes:\n%s", window, out.String())
		}
	}
}

// The cap is what the window leaves beside the first request and the margin, up to 8,192: exact values, so neither a
// fixed cap nor a quarter-of-the-window reserve can come back unnoticed.
func TestClaudeCodeCap(t *testing.T) {
	h := harnesses["claude-code"]
	for window, want := range map[int]int{26000: 3753, 26343: 4096, 28672: 6425, 30439: 8192, 32768: 8192, 65536: 8192, 131072: 8192} {
		if got := h.claudeCodeCap(window); got != want {
			t.Errorf("claudeCodeCap(%d) = %d, want %d", window, got, want)
		}
		if s := claudeCodeSettings("claude-code", h, settings{Window: window, Output: OutputLimit(window)}); s.Output != want {
			t.Errorf("window %d: output %d, want %d", window, s.Output, want)
		}
	}
	if got := h.claudeCodeCap(16384); got < 1024 {
		t.Errorf("claudeCodeCap(16384) = %d: a window under the first request still needs a positive cap", got)
	}
	if s := claudeCodeSettings("codex", h, settings{Window: 32768, Output: 8192}); s.Output != 8192 {
		t.Errorf("codex's output limit was changed to %d", s.Output)
	}
}

// One rule: a cap under 4,096 is warned about, naming -ctx 26343 (the first request, the margin and 4,096), or saying
// the model cannot hold the request when its own maximum is smaller; any other window gets nothing.
func TestClaudeCodeWindowWarning(t *testing.T) {
	for _, c := range []struct {
		window, max int
		want        string // "" no warning, "ctx" one naming -ctx 26343, "cannot" the model cannot hold it
	}{
		{28672, 0, ""}, {32768, 0, ""}, {26343, 0, ""}, {28672, 28672, ""},
		{26342, 0, "ctx"}, {26000, 0, "ctx"}, {16384, 0, "ctx"}, {16384, 32768, "ctx"},
		{16384, 16384, "cannot"}, {26000, 26000, "cannot"},
	} {
		var out bytes.Buffer
		s := testSettings
		s.Window, s.Output, s.MaxPos = c.window, OutputLimit(c.window), c.max
		connect(&out, "claude-code", harnesses["claude-code"], filepath.Join(t.TempDir(), "settings.json"), s, false, false)
		line := ""
		for _, ln := range strings.Split(out.String(), "\n") {
			if strings.Contains(ln, "warning: Claude Code's output cap") {
				line = ln
			}
		}
		got := ""
		switch {
		case strings.Contains(line, "cannot hold") && !strings.Contains(line, "-ctx"):
			got = "cannot"
		case strings.Contains(line, "-ctx 26343 or more"):
			got = "ctx"
		case line != "":
			got = "other: " + line
		}
		if got != c.want || strings.Contains(out.String(), "note:") {
			t.Errorf("window %d (model max %d): %q, want %q\n%s", c.window, c.max, got, c.want, out.String())
		}
	}
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

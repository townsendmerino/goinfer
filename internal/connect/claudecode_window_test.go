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

// The cap the server enforces, as /v1/models publishes it, is what -write gives Claude Code as its window, with the
// output limit computed from that same cap.
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
		if got.Env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] != strconv.Itoa(window) || got.Env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] != strconv.Itoa(OutputLimit(window)) {
			t.Errorf("window %d: wrote %v", window, got.Env)
		}
	}
}

// A served cap that leaves Claude Code less room than claudeCodeHeadroom beside its first request and its output limit
// is warned about, with the smallest -ctx that clears it; a larger cap is not.
func TestClaudeCodeWindowWarning(t *testing.T) {
	warned := func(window int) (bool, string) {
		var out bytes.Buffer
		s := testSettings
		s.Window, s.Output = window, OutputLimit(window)
		connect(&out, "claude-code", harnesses["claude-code"], filepath.Join(t.TempDir(), "settings.json"), s, false, false)
		return strings.Contains(out.String(), "compacts on its own"), out.String()
	}
	// The I3 window, where Claude Code's own auto-compaction thrashed, warns; so do one under the first request and
	// 32,768, whose 8,192 of output limit leaves less than the headroom.
	for _, window := range []int{16384, 28672, 32768} {
		if ok, out := warned(window); !ok {
			t.Errorf("window %d: no warning\n%s", window, out)
		}
	}
	for _, window := range []int{40960, 65536, 131072} {
		if ok, out := warned(window); ok {
			t.Errorf("window %d: warned\n%s", window, out)
		}
	}
	_, out := warned(28672)
	i := strings.Index(out, "-ctx ")
	if i < 0 {
		t.Fatalf("the warning names no -ctx:\n%s", out)
	}
	need, err := strconv.Atoi(strings.Fields(out[i+len("-ctx "):])[0])
	if err != nil {
		t.Fatalf("the -ctx it names is not a number: %v\n%s", err, out)
	}
	if ok, _ := warned(need); ok {
		t.Errorf("-ctx %d, the size the warning asks for, still warns", need)
	}
	if ok, _ := warned(need - 1); !ok {
		t.Errorf("-ctx %d does not warn, so %d is not the smallest that clears it", need-1, need)
	}
	// The warning is Claude Code's alone.
	var other bytes.Buffer
	s := testSettings
	s.Window, s.Output = 8192, OutputLimit(8192)
	connect(&other, "opencode", harnesses["opencode"], filepath.Join(t.TempDir(), "opencode.json"), s, false, false)
	if strings.Contains(other.String(), "compacts on its own") {
		t.Errorf("opencode got Claude Code's warning:\n%s", other.String())
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

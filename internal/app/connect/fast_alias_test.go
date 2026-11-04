package connect

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The server's fast alias (internal/serveapp/fast_alias.go): a served model N advertises "N-fast" in /v1/models, and connect
// points an agent's small-model settings at it. The main model's settings never change, and a server that advertises no alias
// gets the configuration it always did.

var fastSettings = func() settings {
	s := testSettings
	s.Fast = "coder-fast"
	return s
}()

// Claude Code: the two small/fast slots name the alias; every other slot, and the main model, stay the served model.
func TestPatch_claudeCodeSmallModelIsTheFastAlias(t *testing.T) {
	out, err := harnesses["claude-code"].patch(nil, fastSettings)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Env map[string]string `json:"env"`
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	small := map[string]bool{"ANTHROPIC_DEFAULT_HAIKU_MODEL": true, "ANTHROPIC_SMALL_FAST_MODEL": true}
	for _, slot := range append([]string{"ANTHROPIC_MODEL"}, claudeCodeModelSlots...) {
		want := "coder"
		if small[slot] {
			want = "coder-fast"
		}
		if got.Env[slot] != want {
			t.Errorf("env[%s] = %q, want %q", slot, got.Env[slot], want)
		}
	}
	// Without an advertised alias the file is exactly what it was before the alias existed.
	plain, err := harnesses["claude-code"].patch(nil, testSettings)
	if err != nil || strings.Contains(string(plain), "-fast") {
		t.Errorf("a server with no alias got an alias in its settings (%v):\n%s", err, plain)
	}
}

// opencode: small_model names the alias under the goinfer provider, whose model list gains the alias with the main model's limits
// (the alias has no output cap of its own); the main model's entry is unchanged and no `model` is set. An existing small_model is
// replaced and an unrelated key kept.
func TestPatch_opencodeSmallModelIsTheFastAlias(t *testing.T) {
	out, err := harnesses["opencode"].patch([]byte(`{"model": "other/m", "small_model": "other/tiny", "theme": "x"}`), fastSettings)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Model, SmallModel, Theme string
		Provider                 map[string]struct {
			Models map[string]struct {
				Limit struct{ Context, Output int } `json:"limit"`
			} `json:"models"`
		} `json:"provider"`
	}
	raw := strings.NewReplacer(`"small_model"`, `"smallModel"`).Replace(string(out))
	if err := json.Unmarshal([]byte(raw), &got); err != nil {
		t.Fatal(err)
	}
	m := got.Provider["goinfer"].Models
	if got.SmallModel != "goinfer/coder-fast" || got.Model != "other/m" || got.Theme != "x" {
		t.Errorf("small_model %q, model %q, theme %q:\n%s", got.SmallModel, got.Model, got.Theme, out)
	}
	if m["coder"].Limit.Context != 32768 || m["coder"].Limit.Output != 8192 || m["coder-fast"].Limit.Context != 32768 || m["coder-fast"].Limit.Output != 8192 {
		t.Errorf("model limits: main %+v, alias %+v", m["coder"], m["coder-fast"])
	}
	small := fastSettings
	small.Output = 512 // the alias's output limit is the main model's, whatever that is
	if out, _ = harnesses["opencode"].patch(nil, small); strings.Count(string(out), `"output": 512`) != 2 {
		t.Errorf("the alias's output limit is not the main model's:\n%s", out)
	}
	plain, err := harnesses["opencode"].patch(nil, testSettings)
	if err != nil || strings.Contains(string(plain), "-fast") || strings.Contains(string(plain), "small_model") {
		t.Errorf("a server with no alias got an alias in its settings (%v):\n%s", err, plain)
	}
}

// Codex has no small-model setting: the alias changes nothing in its file.
func TestPatch_codexIgnoresTheFastAlias(t *testing.T) {
	a, err1 := harnesses["codex"].patch(nil, testSettings)
	b, err2 := harnesses["codex"].patch(nil, fastSettings)
	if err1 != nil || err2 != nil || !bytes.Equal(a, b) {
		t.Errorf("codex's file depends on the alias (%v, %v):\n%s\n%s", err1, err2, a, b)
	}
}

// -undo removes the alias with the rest, byte for byte, for both agents that carry it.
func TestWriteThenUndoWithFastAlias(t *testing.T) {
	for _, name := range []string{"claude-code", "opencode"} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "cfg")
			orig := []byte("{\n\t\"keep\":   [1,2,3]\n}\n")
			if err := os.WriteFile(path, orig, 0o644); err != nil {
				t.Fatal(err)
			}
			if err := write(harnesses[name], path, fastSettings); err != nil {
				t.Fatal(err)
			}
			if cur, _ := os.ReadFile(path); !strings.Contains(string(cur), "coder-fast") {
				t.Fatalf("the alias was not written:\n%s", cur)
			}
			if err := undo(path, false); err != nil {
				t.Fatal(err)
			}
			if cur, _ := os.ReadFile(path); !bytes.Equal(cur, orig) {
				t.Errorf("after -undo the file is not what it was:\n%q\nwant\n%q", cur, orig)
			}
		})
	}
}

// probe reads the alias from the model's own entry; the alias is not a model of its own in the list, so the single-model
// default still works; a server that advertises none yields none.
func TestProbe_fastAlias(t *testing.T) {
	serve := func(body string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(body)) }))
	}
	for _, tc := range []struct{ name, body, fast string }{
		{"advertised", `{"data":[{"id":"coder","context_window":32768,"fast_model":"coder-fast"}]}`, "coder-fast"},
		{"older server", `{"data":[{"id":"coder","context_window":32768}]}`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ts := serve(tc.body)
			defer ts.Close()
			s, err := probe(ts.Client(), ts.URL, "", "")
			if err != nil || s.Model != "coder" || s.Fast != tc.fast {
				t.Errorf("settings %+v (%v), want model coder, alias %q", s, err, tc.fast)
			}
		})
	}
}

// Through Run: the printed summary names the alias for an agent that has a small-model setting, and the written
// file holds it; for Codex nothing is said.
func TestRun_fastAlias(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"data":[{"id":"coder","context_window":32768,"fast_model":"coder-fast"}]}`))
	}))
	defer ts.Close()
	for _, tc := range []struct {
		agent string
		says  bool
	}{{"claude-code", true}, {"opencode", true}, {"codex", false}} {
		var out bytes.Buffer
		file := filepath.Join(t.TempDir(), "cfg")
		if rc := Run([]string{tc.agent, ts.URL, "-write", "-file", file}, "serve", &out); rc != 0 {
			t.Fatalf("%s: exit %d:\n%s", tc.agent, rc, out.String())
		}
		if got := strings.Contains(out.String(), "coder-fast (the same model, thinking off)"); got != tc.says {
			t.Errorf("%s: summary names the alias = %v, want %v:\n%s", tc.agent, got, tc.says, out.String())
		}
		if cur, _ := os.ReadFile(file); strings.Contains(string(cur), "coder-fast") != tc.says {
			t.Errorf("%s: the file holds the alias = %v, want %v", tc.agent, !tc.says, tc.says)
		}
	}
}

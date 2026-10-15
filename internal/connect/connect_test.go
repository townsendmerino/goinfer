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

var testSettings = settings{URL: "http://127.0.0.1:8080", Key: "goinfer", Model: "coder", Window: 32768, Output: 8192}

// A JSON file keeps its other settings, in the order they were written, and gains only the agent's block.
func TestPatch_claudeCodeKeepsOtherSettings(t *testing.T) {
	old := []byte(`{"theme": "dark", "env": {"EDITOR": "vi", "ANTHROPIC_MODEL": "old"}, "permissions": {"allow": ["Bash(ls:*)"]}}`)
	out, err := harnesses["claude-code"].patch(old, testSettings)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Theme string            `json:"theme"`
		Env   map[string]string `json:"env"`
		Perm  struct {
			Allow []string `json:"allow"`
		} `json:"permissions"`
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("the patched file is not JSON: %v\n%s", err, out)
	}
	want := map[string]string{"EDITOR": "vi", "ANTHROPIC_BASE_URL": "http://127.0.0.1:8080", "ANTHROPIC_AUTH_TOKEN": "goinfer", "ANTHROPIC_MODEL": "coder",
		"ANTHROPIC_DEFAULT_HAIKU_MODEL": "coder", "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "32768", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192", "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"}
	for _, slot := range claudeCodeModelSlots {
		want[slot] = "coder"
	}
	want["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"] = "1"
	if got.Theme != "dark" || len(got.Perm.Allow) != 1 || len(got.Env) != len(want) {
		t.Errorf("other settings were not kept, or env has the wrong keys:\n%s", out)
	}
	for k, v := range want {
		if got.Env[k] != v {
			t.Errorf("env[%s] = %q, want %q", k, got.Env[k], v)
		}
	}
	text := string(out)
	if i, j, k := strings.Index(text, `"theme"`), strings.Index(text, `"env"`), strings.Index(text, `"permissions"`); i < 0 || i >= j || j >= k {
		t.Errorf("the top-level keys were reordered:\n%s", out)
	}
	if e, m := strings.Index(text, `"EDITOR"`), strings.Index(text, `"ANTHROPIC_MODEL"`); e >= m {
		t.Errorf("the env keys were reordered:\n%s", out)
	}
	if again, _ := harnesses["claude-code"].patch(out, testSettings); !bytes.Equal(again, out) {
		t.Errorf("patching a patched file changed it")
	}
}

// opencode: a fresh file gets the schema line and the provider with the served window; an existing provider list keeps
// its other entries; a file with comments is refused, not rewritten without them.
func TestPatch_opencode(t *testing.T) {
	out, err := harnesses["opencode"].patch(nil, testSettings)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Schema   string `json:"$schema"`
		Provider map[string]struct {
			NPM     string `json:"npm"`
			Options struct {
				BaseURL string `json:"baseURL"`
			} `json:"options"`
			Models map[string]struct {
				Limit struct{ Context, Output int } `json:"limit"`
			} `json:"models"`
		} `json:"provider"`
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	p := got.Provider["goinfer"]
	if got.Schema == "" || p.NPM != "@ai-sdk/openai-compatible" || p.Options.BaseURL != "http://127.0.0.1:8080/v1" || p.Models["coder"].Limit.Context != 32768 || p.Models["coder"].Limit.Output != 8192 {
		t.Errorf("a fresh opencode.json:\n%s", out)
	}
	out, err = harnesses["opencode"].patch([]byte(`{"provider": {"other": {"npm": "x"}}, "model": "other/m"}`), testSettings)
	if err != nil || !strings.Contains(string(out), `"other"`) || !strings.Contains(string(out), `"goinfer"`) || !strings.Contains(string(out), `"model": "other/m"`) {
		t.Errorf("an existing provider list was not kept (%v):\n%s", err, out)
	}
	if _, err := harnesses["opencode"].patch([]byte("{\n  // my providers\n  \"provider\": {}\n}\n"), testSettings); err == nil || !strings.Contains(err.Error(), "comments") {
		t.Errorf("a file with comments was patched or refused without saying why: %v", err)
	}
}

// Codex's TOML is patched as text: the top-level keys are set above the first table, an existing goinfer provider table
// is replaced, every other line stays, and a form the patcher cannot see as one block is refused.
func TestPatch_codexTOML(t *testing.T) {
	patch := harnesses["codex"].patch
	fresh, err := patch(nil, testSettings)
	want := "model = \"coder\"\nmodel_provider = \"goinfer\"\nmodel_context_window = 32768\nmodel_auto_compact_token_limit = 24576\n\n" +
		"[model_providers.goinfer]\nname = \"goinfer\"\nbase_url = \"http://127.0.0.1:8080/v1\"\nwire_api = \"responses\"\nenv_key = \"GOINFER_API_KEY\"\n"
	if err != nil || string(fresh) != want {
		t.Errorf("a fresh config.toml (%v):\n%s\nwant:\n%s", err, fresh, want)
	}
	old := "# my codex\nmodel = \"gpt-6\"\napproval_policy = \"on-request\"\n\n[model_providers.goinfer]\nname = \"old\"\nbase_url = \"http://old/v1\"\n\n[profiles.fast]\nmodel = \"small\"\n\n[mcp_servers.docs]\ncommand = \"docs-mcp\"\n"
	out, err := patch([]byte(old), testSettings)
	if err != nil {
		t.Fatal(err)
	}
	got := string(out)
	for _, keep := range []string{"# my codex\n", "approval_policy = \"on-request\"\n", "[profiles.fast]\nmodel = \"small\"\n", "[mcp_servers.docs]\ncommand = \"docs-mcp\"\n"} {
		if !strings.Contains(got, keep) {
			t.Errorf("a line the patch had no business with is gone: %q\n%s", keep, got)
		}
	}
	for _, gone := range []string{"gpt-6", "http://old/v1", "name = \"old\""} {
		if strings.Contains(got, gone) {
			t.Errorf("%q survived the patch:\n%s", gone, got)
		}
	}
	if strings.Count(got, "[model_providers.goinfer]") != 1 || strings.Index(got, "model_provider = \"goinfer\"") > strings.Index(got, "[model_providers.goinfer]") ||
		!strings.Contains(got, "[model_providers.goinfer]\nname = \"goinfer\"\nbase_url = \"http://127.0.0.1:8080/v1\"\nwire_api = \"responses\"\nenv_key = \"GOINFER_API_KEY\"\n\n[profiles.fast]") {
		t.Errorf("the provider table or the top-level keys are misplaced:\n%s", got)
	}
	if again, _ := patch(out, testSettings); !bytes.Equal(again, out) {
		t.Errorf("patching a patched file changed it:\n%s", again)
	}
	for _, bad := range []string{
		"model_providers = { goinfer = { name = \"x\" } }\n",
		"model_providers.goinfer.name = \"x\"\n",
		"[model_providers]\ngoinfer = { name = \"x\" }\n",
	} {
		if _, err := patch([]byte(bad), testSettings); err == nil {
			t.Errorf("a provider written as %q was patched; it cannot be seen as one table", bad)
		}
	}
}

// -write then -undo leaves the file byte for byte as it was, for a file that existed and for one that did not; a second
// -write keeps the first backup; -undo refuses a file changed since, unless forced.
func TestWriteThenUndoIsByteIdentical(t *testing.T) {
	for name, h := range harnesses {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "cfg")
			orig := []byte("{\n\t\"keep\":   [1,2,3]\n}\n")
			if name == "codex" {
				orig = []byte("#   odd   spacing\nmodel   =   \"x\"\n\n\n[profiles.a]\nmodel=\"y\"")
			}
			if err := os.WriteFile(path, orig, 0o644); err != nil {
				t.Fatal(err)
			}
			if err := write(h, path, testSettings); err != nil {
				t.Fatal(err)
			}
			second := testSettings
			second.Window = 16384
			if err := write(h, path, second); err != nil {
				t.Fatal(err)
			}
			if cur, _ := os.ReadFile(path); bytes.Equal(cur, orig) || !strings.Contains(string(cur), "16384") {
				t.Fatalf("the second write did not take:\n%s", cur)
			}
			if err := undo(path, false); err != nil {
				t.Fatal(err)
			}
			if cur, _ := os.ReadFile(path); !bytes.Equal(cur, orig) {
				t.Errorf("after -undo the file is not what it was:\n%q\nwant\n%q", cur, orig)
			}
			if left, _ := filepath.Glob(path + ".goinfer-*"); len(left) != 0 {
				t.Errorf("-undo left %v behind", left)
			}

			fresh := filepath.Join(dir, "sub", "new")
			if err := write(h, fresh, testSettings); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(fresh, []byte("edited since"), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := undo(fresh, false); err == nil || !strings.Contains(err.Error(), "changed since") {
				t.Errorf("-undo over a file edited since: %v, want a refusal", err)
			}
			if err := undo(fresh, true); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(fresh); !os.IsNotExist(err) {
				t.Errorf("-undo of a file that did not exist left it in place (%v)", err)
			}
			if err := undo(fresh, false); err == nil {
				t.Error("-undo with nothing to undo did not say so")
			}
		})
	}
}

// Through Run against a server: the window is read from /v1/models, the configuration is printed without touching
// the file, -write patches it, an agent whose first request does not fit is refused, and several models need -model.
func TestRun(t *testing.T) {
	models := `{"object":"list","data":[{"id":"coder","context_window":32768},{"id":"embed"}]}`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		if models == "auth" && r.Header.Get("Authorization") != "Bearer k" {
			http.Error(w, "no", http.StatusUnauthorized)
			return
		}
		if models == "auth" {
			w.Write([]byte(`{"data":[{"id":"coder","context_window":32768}]}`))
			return
		}
		w.Write([]byte(models))
	}))
	defer ts.Close()
	path := filepath.Join(t.TempDir(), "opencode.json")
	run := func(args ...string) (int, string) {
		var out bytes.Buffer
		return Run(args, "goinfer-serve", &out), out.String()
	}
	rc, out := run("opencode", "-file", path, ts.URL)
	if _, err := os.Stat(path); rc != 0 || !os.IsNotExist(err) || !strings.Contains(out, `"context": 32768`) || !strings.Contains(out, "viable in this window") {
		t.Errorf("printing: rc %d, file exists %v:\n%s", rc, err == nil, out)
	}
	rc, out = run("opencode", "-file", path, "-write", ts.URL+"/v1/")
	if raw, _ := os.ReadFile(path); rc != 0 || !strings.Contains(string(raw), `"baseURL": "`+ts.URL+`/v1"`) {
		t.Errorf("-write: rc %d\n%s\n%s", rc, out, raw)
	}
	if rc, out = run("opencode", "-file", path, "-undo"); rc != 0 {
		t.Errorf("-undo: rc %d\n%s", rc, out)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("-undo left the file it created")
	}

	// Flags after the URL are read, not dropped: -write writes and -undo restores.
	rc, out = run("opencode", ts.URL, "-file", path, "-write")
	if raw, _ := os.ReadFile(path); rc != 0 || !strings.Contains(string(raw), `"baseURL"`) {
		t.Errorf("-write after the URL did not write: rc %d\n%s\n%s", rc, out, raw)
	}
	if rc, out = run("opencode", ts.URL, "-file", path, "-undo"); rc != 0 {
		t.Errorf("-undo after the URL: rc %d\n%s", rc, out)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("-undo after the URL left the file it created")
	}
	if rc, out = run("opencode", ts.URL, "http://other"); rc != 2 || !strings.Contains(out, "unexpected argument") {
		t.Errorf("two URLs: rc %d\n%s", rc, out)
	}

	models = `{"data":[{"id":"coder","context_window":16384}]}`
	cc := filepath.Join(t.TempDir(), "settings.json")
	rc, out = run("claude-code", "-file", cc, "-write", ts.URL)
	if _, err := os.Stat(cc); rc != 1 || !os.IsNotExist(err) || !strings.Contains(out, "not viable") || !strings.Contains(out, "-ctx 25248") {
		t.Errorf("an agent whose first request does not fit was written (rc %d):\n%s", rc, out)
	}
	if rc, _ = run("claude-code", "-file", cc, "-write", "-force", ts.URL); rc != 0 {
		t.Errorf("-force did not write")
	}

	models = `{"data":[{"id":"a","context_window":8192},{"id":"b","context_window":8192}]}`
	if rc, out = run("codex", "-file", cc, ts.URL); rc != 1 || !strings.Contains(out, "choose one with -model") {
		t.Errorf("two models and no -model: rc %d\n%s", rc, out)
	}
	if rc, out = run("codex", "-file", filepath.Join(t.TempDir(), "c.toml"), "-model", "b", ts.URL); rc != 0 || !strings.Contains(out, `model = "b"`) {
		t.Errorf("-model b: rc %d\n%s", rc, out)
	}
	models = "auth"
	if rc, out = run("codex", "-file", cc, ts.URL); rc != 1 || !strings.Contains(out, "-api-key") {
		t.Errorf("a keyed server and no key: rc %d\n%s", rc, out)
	}
	if rc, out = run("codex", "-file", filepath.Join(t.TempDir(), "c.toml"), "-api-key", "k", ts.URL); rc != 0 || !strings.Contains(out, "export GOINFER_API_KEY=k") {
		t.Errorf("a keyed server with its key: rc %d\n%s", rc, out)
	}
	if rc, _ = run("emacs", ts.URL); rc != 2 {
		t.Errorf("an unknown agent: rc %d, want 2", rc)
	}
}

func TestFitAndOutputLimit(t *testing.T) {
	h := harness{FirstRequest: 20000}
	for window, want := range map[int]string{32768: "viable", 25000: "viable", 24999: "marginal", 20001: "marginal", 20000: "not viable", 16384: "not viable"} {
		if got := h.fit(window); got != want {
			t.Errorf("fit(%d) = %q, want %q", window, got, want)
		}
	}
	for window, want := range map[int]int{32768: 8192, 16384: 4096, 4096: 1024, 2048: 1024, 1024: 512, 131072: 8192} {
		if got := OutputLimit(window); got != want {
			t.Errorf("OutputLimit(%d) = %d, want %d", window, got, want)
		}
	}
}

// Every model slot Claude Code's documentation lists (https://code.claude.com/docs/en/model-config § "Environment
// variables" and § "Set a default model for new sessions"; https://code.claude.com/docs/en/env-vars) names the served
// model after -write, including over a stale value, so no alias, subagent or background request names a model the
// server does not serve. The list is written out here rather than read from the code, so a slot dropped there fails here.
func TestWrite_claudeCodeFillsEveryModelSlot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	if err := os.WriteFile(path, []byte(`{"env": {"ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-5-5"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := write(harnesses["claude-code"], path, testSettings); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Env map[string]string `json:"env"`
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("the written file is not JSON: %v\n%s", err, raw)
	}
	for _, slot := range []string{
		"ANTHROPIC_MODEL",                // the session's model
		"ANTHROPIC_DEFAULT_MODEL",        // the model a new session starts on
		"ANTHROPIC_DEFAULT_FABLE_MODEL",  // fable, and best where fable is available
		"ANTHROPIC_DEFAULT_OPUS_MODEL",   // opus, and opusplan in plan mode
		"ANTHROPIC_DEFAULT_SONNET_MODEL", // sonnet, and opusplan outside plan mode
		"ANTHROPIC_DEFAULT_HAIKU_MODEL",  // haiku, and background work
		"CLAUDE_CODE_SUBAGENT_MODEL",     // subagents, teammates and workflow agents not assigned a model
		"ANTHROPIC_SMALL_FAST_MODEL",     // deprecated in favour of ANTHROPIC_DEFAULT_HAIKU_MODEL
	} {
		t.Run(slot, func(t *testing.T) {
			if v, ok := got.Env[slot]; !ok || v != testSettings.Model {
				t.Errorf("env[%s] = %q (set %v), want the served model %q", slot, v, ok, testSettings.Model)
			}
		})
	}
	// A subagent definition that names its own model is overridden too (Claude Code 2.1.257 and later).
	t.Run("CLAUDE_CODE_SUBAGENT_MODEL_FORCE", func(t *testing.T) {
		if v := got.Env["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"]; v != "1" {
			t.Errorf("env[CLAUDE_CODE_SUBAGENT_MODEL_FORCE] = %q, want \"1\"", v)
		}
	})
}

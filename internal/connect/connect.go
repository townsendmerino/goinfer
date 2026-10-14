// Package connect is `goinfer-serve connect`: it writes a coding agent's configuration for a running goinfer server,
// with the context window the server really serves. It reads the server's model list, prints the settings, and with
// -write patches the agent's own file in place, keeping what else is there and leaving a backup that -undo restores
// byte for byte. It never starts the agent and installs nothing into it.
package connect

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// settings is what a configuration is built from: where the server is, which model, and its window.
type settings struct {
	URL    string // the server's base URL, no trailing slash and no /v1
	Key    string // the bearer token the agent sends; "goinfer" when the server has none
	Model  string
	Window int // context_window from /v1/models
	Output int // the output limit told to the agent
}

// harness is one coding agent connect can configure.
type harness struct {
	Name string
	// FirstRequest is the size in tokens of the agent's first request, its system prompt and tool definitions, as
	// measured on the version named (docs/measurements/integration-harness-2026-10/harness-census-2026-10-12.md).
	FirstRequest int
	Version      string
	// path is the file -write patches. global picks the per-user file where the agent also has a per-project one.
	path func(home, dir string, global bool) string
	// patch returns the file's new contents given its old ones (empty for a file that does not exist).
	patch func(old []byte, s settings) ([]byte, error)
	// after is what the person still has to do: the environment to set, the command to run.
	after func(s settings) string
}

var harnesses = map[string]harness{
	"claude-code": {
		Name: "Claude Code", FirstRequest: 20199, Version: "2.1.296",
		path: func(home, _ string, _ bool) string { return filepath.Join(home, ".claude", "settings.json") },
		patch: func(old []byte, s settings) ([]byte, error) {
			return patchJSON(old, func(o *object) error {
				env, err := o.get("env")
				if err != nil {
					return err
				}
				// The background model is named so the session-title request does not go looking for a model the
				// server does not serve.
				for _, kv := range [][2]string{{"ANTHROPIC_BASE_URL", s.URL}, {"ANTHROPIC_AUTH_TOKEN", s.Key}, {"ANTHROPIC_MODEL", s.Model},
					{"ANTHROPIC_DEFAULT_HAIKU_MODEL", s.Model}, {"CLAUDE_CODE_MAX_CONTEXT_TOKENS", strconv.Itoa(s.Window)},
					// Claude Code asks for 32,000 output tokens on every request and counts them against the window it
					// is told, so at 32,768 it sees no room and tries to compact at once.
					{"CLAUDE_CODE_MAX_OUTPUT_TOKENS", strconv.Itoa(s.Output)},
					// Claude Code opens its system prompt with a hash of the conversation's first message, at token 22, so a prefix
					// match between two conversations ends there. Off, they share the preamble before the user's own text. The
					// server does not rewrite the line: this is the client's setting (docs/integrations/claude-code.md).
					{"CLAUDE_CODE_ATTRIBUTION_HEADER", "0"}} {
					env.set(kv[0], kv[1])
				}
				o.set("env", env)
				return nil
			})
		},
		after: func(settings) string { return "run: claude" },
	},
	"opencode": {
		Name: "opencode", FirstRequest: 7165, Version: "1.19.0",
		path: func(home, dir string, global bool) string {
			if global {
				return filepath.Join(home, ".config", "opencode", "opencode.json")
			}
			return filepath.Join(dir, "opencode.json")
		},
		patch: func(old []byte, s settings) ([]byte, error) {
			return patchJSON(old, func(o *object) error {
				if len(*o) == 0 {
					o.set("$schema", "https://opencode.ai/config.json")
				}
				prov, err := o.get("provider")
				if err != nil {
					return err
				}
				// Without limit.context opencode never compacts, and runs the conversation into the window.
				prov.set("goinfer", map[string]any{
					"npm": "@ai-sdk/openai-compatible", "name": "goinfer",
					"options": map[string]any{"baseURL": s.URL + "/v1", "apiKey": s.Key},
					"models":  map[string]any{s.Model: map[string]any{"name": s.Model, "limit": map[string]any{"context": s.Window, "output": s.Output}}},
				})
				o.set("provider", prov)
				return nil
			})
		},
		after: func(s settings) string { return "run: opencode -m goinfer/" + s.Model }},
	"codex": {
		Name: "Codex", FirstRequest: 5850, Version: "0.162.1",
		path: func(home, _ string, _ bool) string {
			if ch := os.Getenv("CODEX_HOME"); ch != "" {
				return filepath.Join(ch, "config.toml")
			}
			return filepath.Join(home, ".codex", "config.toml")
		},
		patch: func(old []byte, s settings) ([]byte, error) {
			q := strconv.Quote
			out, err := tomlEdit(string(old),
				[]tomlRow{{"model", q(s.Model)}, {"model_provider", q("goinfer")}, {"model_context_window", strconv.Itoa(s.Window)},
					{"model_auto_compact_token_limit", strconv.Itoa(s.Window - s.Output)}},
				"model_providers.goinfer",
				// wire_api is "responses": Codex rejects "chat" as a configuration error.
				[]tomlRow{{"name", q("goinfer")}, {"base_url", q(s.URL + "/v1")}, {"wire_api", q("responses")}, {"env_key", q("GOINFER_API_KEY")}})
			return []byte(out), err
		},
		after: func(s settings) string {
			return "Codex reads the key from the environment: export GOINFER_API_KEY=" + s.Key + "\nrun: codex"
		},
	},
}

// patchJSON applies edit to a JSON object file and returns it indented, with the keys it had in the order it had them.
func patchJSON(old []byte, edit func(*object) error) ([]byte, error) {
	var o object
	if len(bytes.TrimSpace(old)) > 0 {
		var err error
		if o, err = parseObject(old); err != nil {
			return nil, fmt.Errorf("it is not plain JSON (%v); a file with comments cannot be patched without losing them", err)
		}
	}
	if err := edit(&o); err != nil {
		return nil, err
	}
	return o.text(), nil
}

// OutputLimit is the output limit told to an agent: a quarter of the window, at least 1,024 and at most 8,192 (and never
// more than half a small window). serve gives the same limit to a /v1/responses request that names none, so an agent
// that sends no cap of its own gets the one its configuration was written with.
func OutputLimit(window int) int { return min(max(window/4, 1024), 8192, max(window/2, 1)) }

// fit says whether an agent's first request fits a window: "viable" with a quarter to spare, "marginal" when only the
// request itself fits, "not viable" otherwise.
func (h harness) fit(window int) string {
	switch {
	case h.FirstRequest+h.FirstRequest/4 <= window:
		return "viable"
	case h.FirstRequest < window:
		return "marginal"
	}
	return "not viable"
}

// probe reads the server's model list and returns the chosen model's settings. model "" takes the only model, and is
// an error when there are several.
func probe(c *http.Client, url, key, model string) (settings, error) {
	req, err := http.NewRequest("GET", url+"/v1/models", nil)
	if err != nil {
		return settings{}, err
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := c.Do(req)
	if err != nil {
		return settings{}, fmt.Errorf("no goinfer server answered at %s: %v", url, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != 200 {
		return settings{}, fmt.Errorf("GET %s/v1/models: %s (pass -api-key if the server was started with one)", url, resp.Status)
	}
	var list struct {
		Data []struct {
			ID     string `json:"id"`
			Window int    `json:"context_window"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &list); err != nil {
		return settings{}, fmt.Errorf("GET %s/v1/models: not a model list: %v", url, err)
	}
	var ids []string
	for _, m := range list.Data {
		if m.Window > 0 {
			ids = append(ids, m.ID)
		}
	}
	sort.Strings(ids)
	if model == "" {
		if len(ids) != 1 {
			return settings{}, fmt.Errorf("the server serves %d generative models (%s); choose one with -model", len(ids), strings.Join(ids, ", "))
		}
		model = ids[0]
	}
	for _, m := range list.Data {
		if m.ID == model && m.Window > 0 {
			s := settings{URL: url, Key: key, Model: model, Window: m.Window, Output: OutputLimit(m.Window)}
			if s.Key == "" {
				s.Key = "goinfer" // agents refuse an empty key; a server with none accepts any
			}
			return s, nil
		}
	}
	return settings{}, fmt.Errorf("the server does not serve a generative model named %q (it serves: %s)", model, strings.Join(ids, ", "))
}

// restoreRecord is what -write leaves beside a patched file so -undo can put it back: whether the file existed, and
// the hash of what -write left, so an edit made since is noticed.
type restoreRecord struct {
	Path    string `json:"path"`
	Existed bool   `json:"existed"`
	Written string `json:"written_sha256"`
	When    string `json:"when"`
}

func backupPath(p string) string { return p + ".goinfer-backup" }
func recordPath(p string) string { return p + ".goinfer-restore.json" }

func sum(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

// write patches path with h's settings. The first write saves the original bytes; a later one keeps that first backup,
// so -undo always returns to the file as it was before goinfer touched it.
func write(h harness, path string, s settings) error {
	old, err := os.ReadFile(path)
	existed := err == nil
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	next, err := h.patch(old, s)
	if err != nil {
		return fmt.Errorf("%s: %v", path, err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if _, err := os.Stat(recordPath(path)); os.IsNotExist(err) {
		if existed {
			if err := os.WriteFile(backupPath(path), old, 0o600); err != nil {
				return err
			}
		}
	} else if err != nil {
		return err
	} else if prev, rerr := readRecord(path); rerr == nil {
		existed = prev.Existed
	}
	rec, _ := json.MarshalIndent(restoreRecord{Path: path, Existed: existed, Written: sum(next), When: time.Now().UTC().Format(time.RFC3339)}, "", "  ")
	if err := os.WriteFile(recordPath(path), append(rec, '\n'), 0o600); err != nil {
		return err
	}
	return os.WriteFile(path, next, 0o600)
}

func readRecord(path string) (restoreRecord, error) {
	var rec restoreRecord
	raw, err := os.ReadFile(recordPath(path))
	if err != nil {
		return rec, err
	}
	return rec, json.Unmarshal(raw, &rec)
}

// undo restores path to what it was before the first -write: its original bytes, or no file. It refuses when the
// file has changed since goinfer wrote it, unless force.
func undo(path string, force bool) error {
	rec, err := readRecord(path)
	if os.IsNotExist(err) {
		return fmt.Errorf("%s was not written by `connect -write` (no %s)", path, filepath.Base(recordPath(path)))
	}
	if err != nil {
		return err
	}
	cur, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if !force && sum(cur) != rec.Written {
		return fmt.Errorf("%s has changed since `connect -write`; -undo would discard that change (-force restores anyway)", path)
	}
	if rec.Existed {
		orig, err := os.ReadFile(backupPath(path))
		if err != nil {
			return fmt.Errorf("the backup is missing: %v", err)
		}
		if err := os.WriteFile(path, orig, 0o600); err != nil {
			return err
		}
	} else if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	_ = os.Remove(backupPath(path))
	return os.Remove(recordPath(path))
}

const usage = `%[1]s connect — write a coding agent's configuration for a running goinfer server

  %[1]s connect <agent> [url]      agent: %[2]s; url default http://127.0.0.1:8080

Prints the configuration, with the context window the server serves. -write patches the agent's own
file in place, keeping its other settings, and leaves a backup; -undo restores that file byte for byte.
It never starts the agent.

`

// Run implements `serve connect`. args excludes the program name and the "connect" word.
func Run(args []string, self string, out io.Writer) int {
	names := make([]string, 0, len(harnesses))
	for n := range harnesses {
		names = append(names, n)
	}
	sort.Strings(names)
	fs := flag.NewFlagSet("connect", flag.ContinueOnError)
	fs.SetOutput(out)
	fs.Usage = func() { fmt.Fprintf(out, usage, self, strings.Join(names, ", ")); fs.PrintDefaults() }
	model := fs.String("model", "", "served model id (default: the only generative model the server serves)")
	apiKey := fs.String("api-key", "", "bearer token, if the server was started with -api-key")
	doWrite := fs.Bool("write", false, "patch the agent's configuration file in place (default: print the configuration)")
	doUndo := fs.Bool("undo", false, "restore the file -write patched to what it was before")
	force := fs.Bool("force", false, "with -write: write although the agent's first request does not fit the window; with -undo: restore although the file changed since")
	global := fs.Bool("global", false, "opencode: the per-user file (~/.config/opencode/opencode.json) instead of ./opencode.json")
	file := fs.String("file", "", "the configuration file to patch or restore, instead of the agent's usual one")
	// The agent name comes first, flags after it, as in `connect codex -write`.
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		fs.Usage()
		return 2
	}
	h, ok := harnesses[args[0]]
	if !ok {
		fmt.Fprintf(out, "connect: unknown agent %q (known: %s)\n", args[0], strings.Join(names, ", "))
		return 2
	}
	// Flags may follow the URL (`connect codex http://… -write`): package flag stops at the first operand, and a
	// -write or -undo left unparsed after it would silently print the configuration instead.
	var operands []string
	for rest := args[1:]; ; {
		if err := fs.Parse(rest); err != nil {
			return 2
		}
		if fs.NArg() == 0 {
			break
		}
		operands = append(operands, fs.Arg(0))
		rest = fs.Args()[1:]
	}
	if len(operands) > 1 {
		fmt.Fprintf(out, "connect: unexpected argument %q (one server URL at most)\n", operands[1])
		return 2
	}
	url := "http://127.0.0.1:8080"
	if len(operands) > 0 {
		url = operands[0]
	}
	if !strings.Contains(url, "://") {
		url = "http://" + url
	}
	url = strings.TrimSuffix(strings.TrimSuffix(strings.TrimRight(url, "/"), "/v1"), "/")
	home, _ := os.UserHomeDir()
	dir, _ := os.Getwd()
	path := h.path(home, dir, *global)
	if *file != "" {
		path = *file
	}
	if *doUndo {
		if err := undo(path, *force); err != nil {
			fmt.Fprintf(out, "connect: %v\n", err)
			return 1
		}
		fmt.Fprintf(out, "restored %s\n", path)
		return 0
	}
	s, err := probe(&http.Client{Timeout: 10 * time.Second}, url, *apiKey, *model)
	if err != nil {
		fmt.Fprintf(out, "connect: %v\n", err)
		return 1
	}
	return connect(out, args[0], h, path, s, *doWrite, *force)
}

// connect prints the configuration for s and, with doWrite, patches path. It refuses to write a configuration whose
// agent cannot fit its first request in the window, unless force.
func connect(out io.Writer, name string, h harness, path string, s settings, doWrite, force bool) int {
	verdict := h.fit(s.Window)
	fmt.Fprintf(out, "%s -> %s, model %s, context window %d tokens (output limit %d)\n", h.Name, s.URL, s.Model, s.Window, s.Output)
	fmt.Fprintf(out, "first request: %d tokens (%s %s, measured) — %s in this window\n", h.FirstRequest, h.Name, h.Version, verdict)
	switch verdict {
	case "marginal":
		fmt.Fprintf(out, "  the first request fits, with less than a quarter of it to spare: the session will compact almost at once\n")
	case "not viable":
		fmt.Fprintf(out, "  the first request does not fit: restart goinfer-serve with -ctx %d or more\n", h.FirstRequest+h.FirstRequest/4)
	}
	fmt.Fprint(out, claudeCodeWarnings(name, s))
	old, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		fmt.Fprintf(out, "connect: %v\n", err)
		return 1
	}
	next, perr := h.patch(old, s)
	if !doWrite {
		if perr != nil {
			// The file cannot be patched; show what a fresh one would hold, for pasting.
			fmt.Fprintf(out, "\n%s cannot be patched: %v\nthe settings, for a file of their own:\n\n", path, perr)
			if next, err = h.patch(nil, s); err != nil {
				fmt.Fprintf(out, "connect: %v\n", err)
				return 1
			}
		} else {
			fmt.Fprintf(out, "\n%s would hold (-write patches it, -undo restores it):\n\n", path)
		}
		out.Write(next)
		fmt.Fprintf(out, "\n%s\n", h.after(s))
		return 0
	}
	if perr != nil {
		fmt.Fprintf(out, "connect: %s: %v\n", path, perr)
		return 1
	}
	if verdict == "not viable" && !force {
		fmt.Fprintf(out, "connect: not written (-force writes it anyway)\n")
		return 1
	}
	if err := write(h, path, s); err != nil {
		fmt.Fprintf(out, "connect: %v\n", err)
		return 1
	}
	fmt.Fprintf(out, "\nwrote %s (`connect %s -undo` restores it)\n%s\n", path, name, h.after(s))
	return 0
}

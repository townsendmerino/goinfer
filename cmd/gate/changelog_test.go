package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const serveSrc = `package serveapp

func (s *server) routes(mux *mux) {
	// mux.HandleFunc("GET /v1/commented", nil)
	mux.HandleFunc("POST /v1/chat/completions", s.chat)
	mux.HandleFunc("GET /v1/jobs/{id}", s.job)
	mux.Handle("GET /health", s.health)
	other.Register("POST /v1/not-a-route", nil)
}
`

func writeTree(t *testing.T, root string, files map[string]string) {
	t.Helper()
	for p, c := range files {
		full := filepath.Join(root, p)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(c), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func baseFiles() map[string]string {
	return map[string]string{
		"internal/serveapp/routes.go":      serveSrc,
		"internal/serveapp/routes_test.go": `package serveapp; func x(m *mux) { m.HandleFunc("GET /v1/only-in-a-test", nil) }`,
		"docs/flags.meta.json":             `{"flags":[{"name":"model"},{"name":"old-flag"}]}`,
		"docs/capability-matrix.json":      `[{"name":"llama","display_name":"Llama","model_types":["llama"]}]`,
		"go.mod":                           "module x\n\nrequire (\n\tgithub.com/townsendmerino/aikit v1.60.0\n)\n",
	}
}

func TestServeRoutes_readsRegistrationsNotCommentsOrTests(t *testing.T) {
	root := t.TempDir()
	writeTree(t, root, baseFiles())
	got, err := serveRoutes(dirSource{root})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"POST /v1/chat/completions", "GET /v1/jobs/{}", "GET /health"}
	if len(got) != len(want) {
		t.Fatalf("routes = %v, want %v", got, want)
	}
	for _, w := range want {
		if !got[w] {
			t.Errorf("missing %q in %v", w, got)
		}
	}
}

// The scan is not allowed to read a refactor as "no routes": serve registers dozens today.
func TestServeRoutes_findsTheRealOnes(t *testing.T) {
	root, err := repoRoot()
	if err != nil {
		t.Skip(err)
	}
	got, err := serveRoutes(dirSource{root})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) < 30 || !got["POST /v1/chat/completions"] || !got["GET /v1/models"] {
		t.Fatalf("the route scan found %d routes (%v): it is not reading serve's registrations", len(got), got)
	}
}

func noWaive(kind, name string) bool { return false }

func states(t *testing.T) (base, head *clState) {
	t.Helper()
	root := t.TempDir()
	writeTree(t, root, baseFiles())
	b, err := readClState(dirSource{root})
	if err != nil {
		t.Fatal(err)
	}
	h2 := baseFiles()
	h2["internal/serveapp/routes.go"] = serveSrc + `
func (s *server) more(mux *mux) {
	mux.HandleFunc("POST /v1/audio/transcriptions", s.tr)
	mux.HandleFunc("POST /v1/batches", s.b)
	mux.HandleFunc("GET /v1/batches/{id}", s.gb)
	mux.HandleFunc("POST /v1/batches/{name}/cancel", s.cb)
}
`
	h2["docs/flags.meta.json"] = `{"flags":[{"name":"model"},{"name":"windowed-kv"}]}`
	h2["docs/capability-matrix.json"] = `[{"name":"llama","display_name":"Llama","model_types":["llama"]},{"name":"qwen3_asr","display_name":"Qwen3-ASR","model_types":["qwen3_asr"]}]`
	h2["go.mod"] = "module x\n\nrequire (\n\tgithub.com/townsendmerino/aikit v1.69.0\n)\n"
	root2 := t.TempDir()
	writeTree(t, root2, h2)
	h, err := readClState(dirSource{root2})
	if err != nil {
		t.Fatal(err)
	}
	return b, h
}

func owedSet(o []owedItem) string {
	var s []string
	for _, i := range o {
		s = append(s, i.Kind+":"+i.Name)
	}
	return strings.Join(s, "|")
}

// Everything new is owed against an empty section, and each line added clears exactly its own item.
func TestChangelogOwed_eachKindAndEachLineClearsOnlyItsItem(t *testing.T) {
	base, head := states(t)
	all := owedSet(changelogOwed(base, head, "", noWaive))
	for _, want := range []string{"route:POST /v1/audio/transcriptions", "route:POST /v1/batches", "flag:windowed-kv", "flag:old-flag", "family:qwen3_asr", "pin:aikit v1.69.0"} {
		if !strings.Contains(all, want) {
			t.Errorf("owed set %q lacks %q", all, want)
		}
	}
	lines := map[string]string{
		"route:POST /v1/audio/transcriptions": "the /v1/audio/transcriptions route",
		"flag:windowed-kv":                    "`--windowed-kv` is on",
		"flag:old-flag":                       "removed `--old-flag`",
		"family:qwen3_asr":                    "Qwen3-ASR is served",
		"pin:aikit v1.69.0":                   "aikit v1.69.0 pin",
		"route:POST /v1/batches":              "the /v1/batches routes",
	}
	for item, line := range lines {
		got := owedSet(changelogOwed(base, head, line, noWaive))
		if strings.Contains(got, item) {
			t.Errorf("the line %q did not clear %s (still owed: %s)", line, item, got)
		}
		if n := strings.Count(all, "|") - strings.Count(got, "|"); n < 0 || (n == 0 && item != "route:POST /v1/batches") {
			t.Errorf("the line %q for %s changed the owed set from %q to %q", line, item, all, got)
		}
	}
	full := strings.Join([]string{"/v1/audio/transcriptions", "--windowed-kv", "--old-flag", "Qwen3-ASR", "v1.69.0", "/v1/batches"}, "\n")
	if got := changelogOwed(base, head, full, noWaive); len(got) != 0 {
		t.Errorf("a section naming every change still owes %v", got)
	}
}

// A new family's sub-routes ride with the family route, with placeholders read as {}; an unrelated mention does not cover them.
func TestChangelogOwed_routePrefixAndPlaceholders(t *testing.T) {
	base, head := states(t)
	got := owedSet(changelogOwed(base, head, "`POST /v1/batches` and the rest", noWaive))
	if strings.Contains(got, "batches") {
		t.Errorf("sub-routes of the new /v1/batches were owed though the family route is named: %s", got)
	}
	got = owedSet(changelogOwed(base, head, "/v1/audio is mentioned and /v1/chat/completions too", noWaive))
	if !strings.Contains(got, "route:POST /v1/audio/transcriptions") {
		t.Errorf("a parent-path mention covered a route: %s", got)
	}
	if got := owedSet(changelogOwed(base, head, "/v1/batches/{id}/cancel", noWaive)); strings.Contains(got, "cancel") {
		t.Errorf("a placeholder spelled differently was not read as {}: %s", got)
	}
}

func TestChangelogOwed_waiversClearAndAreNeeded(t *testing.T) {
	base, head := states(t)
	w := func(kind, name string) bool { return kind == "pin" }
	if got := owedSet(changelogOwed(base, head, "", w)); strings.Contains(got, "pin:") {
		t.Errorf("a waived pin is still owed: %s", got)
	}
	dir := t.TempDir()
	p := filepath.Join(dir, "w.txt")
	if err := os.WriteFile(p, []byte("# c\nroute POST /admin/x internal, not a user surface\nflag foo a debugging flag\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	ws, err := loadChangelogWaivers(p)
	if err != nil || ws["route POST /admin/x"] == "" || ws["flag foo"] == "" {
		t.Fatalf("waivers = %v, %v", ws, err)
	}
	for _, bad := range []string{"route POST /admin/x\n", "route POST /admin/x\nflag foo\n", "flag foo\n"} {
		if err := os.WriteFile(p, []byte(bad), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := loadChangelogWaivers(p); err == nil {
			t.Errorf("a waiver with no reason was accepted: %q", bad)
		}
	}
}

func TestClSection(t *testing.T) {
	doc := "# C\n\n## [Unreleased]\nnew things\n\n## [v0.2.0] — d\nold\n"
	if s, ok := clSection(doc, "Unreleased"); !ok || !strings.Contains(s, "new things") || strings.Contains(s, "old") {
		t.Errorf("Unreleased = %q, %v", s, ok)
	}
	if _, ok := clSection(doc, "v0.9.9"); ok {
		t.Error("an absent section was found")
	}
}

// Through git: the base side is read with git show at a tag, the head side from the tree, and a route added after the tag is owed.
func TestRunChangelog_throughAGitRepo(t *testing.T) {
	dir, _ := gpuTestRepo(t) // a repository with an initial commit
	writeTree(t, dir, baseFiles())
	writeTree(t, dir, map[string]string{"CHANGELOG.md": "## [Unreleased]\n\n## [v0.1.0] — d\n"})
	gitIn(t, dir, "add", ".")
	gitIn(t, dir, "commit", "-q", "-m", "release")
	gitIn(t, dir, "tag", "v0.1.0")
	head := baseFiles()
	head["internal/serveapp/routes.go"] = serveSrc + "func (s *server) m(mux *mux) { mux.HandleFunc(\"POST /v1/new\", nil) }\n"
	writeTree(t, dir, head)
	gitIn(t, dir, "add", ".")
	gitIn(t, dir, "commit", "-q", "-m", "a route")
	bs, err := readClState(gitSource{dir, "v0.1.0"})
	if err != nil {
		t.Fatal(err)
	}
	hs, err := readClState(dirSource{dir})
	if err != nil {
		t.Fatal(err)
	}
	if bs.Routes["POST /v1/new"] || !hs.Routes["POST /v1/new"] {
		t.Fatalf("base has the route %v, head has it %v", bs.Routes["POST /v1/new"], hs.Routes["POST /v1/new"])
	}
	if got := owedSet(changelogOwed(bs, hs, "", noWaive)); got != "route:POST /v1/new" {
		t.Errorf("owed = %q, want only the new route", got)
	}
}

// TestChangelogFragments: the Unreleased entries come from changelog.d/, newest name first, README.md skipped, and a tree
// without the directory has none.
func TestChangelogFragments(t *testing.T) {
	root := t.TempDir()
	if got, err := changelogFragments(dirSource{root}); err != nil || got != "" {
		t.Fatalf("no changelog.d: got %q, %v; want none", got, err)
	}
	dir := filepath.Join(root, changelogFragmentDir)
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{
		"README.md":                "how to add one\n",
		"2026-10-20-01-older.md":   "- **older** entry\n",
		"2026-10-27-newer-flag.md": "- **newer**: `--new-flag`\n",
		"2026-10-21-01-between.md": "- **between**\n  two lines\n",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got, err := changelogFragments(dirSource{root})
	if err != nil {
		t.Fatal(err)
	}
	want := "- **newer**: `--new-flag`\n- **between**\n  two lines\n- **older** entry\n"
	if got != want {
		t.Errorf("fragments = %q, want %q", got, want)
	}
}

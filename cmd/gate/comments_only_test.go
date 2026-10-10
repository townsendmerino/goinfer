package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The comments-only gate is only worth having if it can go red. Each case below is one way a "comments commit" could carry code, plus
// the comment rewrites it must wave through; the repo-level cases run the CLI path against a scratch git repository.

const coBase = `//go:build linux

// Package p does things.
package p

import "fmt"

// Greet says hello.
//
// It took three tries in 2026-09 to get the spacing right; see docs/x.md.
func Greet(name string) string {
	a := 1

	b := 2
	return fmt.Sprintf("hi %s %d", name, a+b) // trailing
}

const kernel = ` + "`__kernel void f() { x[0] = 1; }`" + `

//go:embed none.txt
var _ = 0

//nolint:unused
func unused() {}
`

const coExample = `package p

import "fmt"

func ExampleGreet() {
	// the call
	fmt.Println(Greet("a"))
	// Output: hi a 3
}
`

func TestCommentsOnly_commentRewritesAreGreen(t *testing.T) {
	cases := map[string]string{
		"doc comment rewritten":       strings.Replace(coBase, "// It took three tries in 2026-09 to get the spacing right; see docs/x.md.", "// Spacing is deliberate; history in docs/code-notes/p.md#Greet.", 1),
		"doc comment deleted":         strings.Replace(coBase, "// Greet says hello.\n//\n// It took three tries in 2026-09 to get the spacing right; see docs/x.md.\n", "", 1),
		"trailing comment deleted":    strings.Replace(coBase, " // trailing", "", 1),
		"comment added between stmts": strings.Replace(coBase, "\ta := 1\n", "\ta := 1\n\t// new note\n", 1),
		// go/printer would keep a blank line where the comment was and print the two files differently; a token stream does not.
		"comment line between statements deleted": strings.Replace(coBase, "\ta := 1\n\n\tb := 2", "\ta := 1\n\t// to go\n\tb := 2", 1),
	}
	for name, cur := range cases {
		v := compareComments("p.go", []byte(coBase), []byte(cur))
		if len(v.reds) != 0 {
			t.Errorf("%s: want green, got %v", name, v.reds)
		}
	}
	// the printer pitfall, as its own pair: a comment line between two statements, then deleted
	with := strings.Replace(coBase, "\ta := 1\n\n\tb := 2", "\ta := 1\n\t// to go\n\tb := 2", 1)
	without := strings.Replace(coBase, "\ta := 1\n\n\tb := 2", "\ta := 1\n\tb := 2", 1)
	if v := compareComments("p.go", []byte(with), []byte(without)); len(v.reds) != 0 {
		t.Errorf("deleting a comment line between statements: want green, got %v", v.reds)
	}
}

func TestCommentsOnly_codeChangesAreRed(t *testing.T) {
	cases := map[string]string{
		"identifier renamed":      strings.ReplaceAll(coBase, "Greet", "Hello"),
		"string literal changed":  strings.Replace(coBase, "hi %s %d", "hey %s %d", 1),
		"kernel string changed":   strings.Replace(coBase, "x[0] = 1", "x[0] = 2", 1),
		"statement added":         strings.Replace(coBase, "\tb := 2\n", "\tb := 2\n\t_ = b\n", 1),
		"go:build removed":        strings.Replace(coBase, "//go:build linux\n\n", "", 1),
		"go:build edited":         strings.Replace(coBase, "//go:build linux", "//go:build linux && amd64", 1),
		"go:embed edited":         strings.Replace(coBase, "//go:embed none.txt", "//go:embed other.txt", 1),
		"nolint edited":           strings.Replace(coBase, "//nolint:unused", "//nolint:deadcode", 1),
		"function removed":        strings.Replace(coBase, "func unused() {}\n", "", 1),
		"comment swallowed code":  strings.Replace(coBase, "\ta := 1\n", "\t// a := 1\n", 1),
		"statement commented out": strings.Replace(coBase, "\tb := 2\n", "\t// b := 2\n", 1),
	}
	for name, cur := range cases {
		v := compareComments("p.go", []byte(coBase), []byte(cur))
		if len(v.reds) == 0 {
			t.Errorf("%s: want red, got green", name)
		}
	}
}

func TestCommentsOnly_exampleComments(t *testing.T) {
	// a comment inside an Example is protected: its `// Output:` is the test's expectation
	edit := strings.Replace(coExample, "// Output: hi a 3", "// Output: hi a 4", 1)
	if v := compareComments("p_test.go", []byte(coExample), []byte(edit)); len(v.reds) == 0 || !strings.Contains(v.reds[0], "PROTECTED") {
		t.Errorf("Output edit: want a PROTECTED COMMENT red, got %v", v.reds)
	}
	edit = strings.Replace(coExample, "// the call", "// the call, reworded", 1)
	if v := compareComments("p_test.go", []byte(coExample), []byte(edit)); len(v.reds) == 0 {
		t.Errorf("ordinary comment inside an Example: want red, got green")
	}
	// the same edit in a function that is not an Example, and an Example's own doc comment, are ordinary comments
	notExample := strings.Replace(coExample, "func ExampleGreet", "func helper", 1)
	edit = strings.Replace(notExample, "// the call", "// reworded", 1)
	if v := compareComments("p_test.go", []byte(notExample), []byte(edit)); len(v.reds) != 0 {
		t.Errorf("comment inside a non-Example: want green, got %v", v.reds)
	}
	documented := strings.Replace(coExample, "func ExampleGreet", "// ExampleGreet shows Greet.\nfunc ExampleGreet", 1)
	if v := compareComments("p_test.go", []byte(documented), []byte(coExample)); len(v.reds) != 0 {
		t.Errorf("an Example's doc comment: want green, got %v", v.reds)
	}
	// a file that is not a _test.go has no Examples
	edit = strings.Replace(coExample, "// the call", "// the call, reworded", 1)
	if v := compareComments("p.go", []byte(coExample), []byte(edit)); len(v.reds) != 0 {
		t.Errorf("Example-named func outside a test file: want green, got %v", v.reds)
	}
}

func TestCommentsOnly_rewrapThatCreatesADirectiveLookalikeIsRed(t *testing.T) {
	cur := strings.Replace(coBase, "// Greet says hello.", "// Greet says hello and\n//     go:embed list is not a directive.", 1)
	v := compareComments("p.go", []byte(coBase), []byte(cur))
	if len(v.reds) == 0 || !strings.Contains(v.reds[0], "SA9009") {
		t.Errorf("want an SA9009 red, got %v", v.reds)
	}
	// the same line already in the base file is not this diff's doing
	if v := compareComments("p.go", []byte(cur), []byte(cur+"\n// more\n")); len(v.reds) != 0 {
		t.Errorf("a lookalike already present before: want green, got %v", v.reds)
	}
}

func TestCommentsOnly_unparseableIsRedNotAPanic(t *testing.T) {
	v := compareComments("p.go", []byte(coBase), []byte(coBase+"\nfunc broken( {\n"))
	if len(v.reds) == 0 {
		t.Fatal("a file that does not parse must be red")
	}
}

func TestCommentsOnly_movedTextReport(t *testing.T) {
	story := "The 2026-09-12 debugging story: two causes, found by bisecting."
	old := "package p\n\n// Run does it.\n//\n// " + story + "\n// Another thing that was said.\nfunc Run() {}\n"
	cur := "package p\n\n// Run does it. History: docs/code-notes/p.md#Run.\nfunc Run() {}\n"
	v := compareComments("p.go", []byte(old), []byte(cur))
	if len(v.reds) != 0 {
		t.Fatalf("comment-only: %v", v.reds)
	}
	if len(v.removed) != 1 || v.removed[0].lines != 3 {
		t.Fatalf("want one removed run of 3 lines (\"Run does it.\" is gone as a line too), got %+v", v.removed)
	}
	// verbatim in a blockquote: found
	r := movedTextReport(v.removed, "## Run\n\n> Run does it.\n>\n> "+story+"\n> Another thing that was said.\n")
	if len(r.found) != 1 || len(r.missing) != 0 {
		t.Errorf("verbatim move: want found, got found=%d missing=%d", len(r.found), len(r.missing))
	}
	// edited on the way: listed, not found
	r = movedTextReport(v.removed, "## Run\n\nThe story, summarised.\n")
	if len(r.missing) != 1 {
		t.Errorf("unmoved text: want listed as missing, got %+v", r)
	}
	var out bytes.Buffer
	r.write(&out, false)
	if !strings.Contains(out.String(), "NOT FOUND") || !strings.Contains(out.String(), "p.go:") {
		t.Errorf("report should name the file and the run:\n%s", out.String())
	}
}

// ---- the CLI path, against a scratch repository ----

func coRepo(t *testing.T) string {
	t.Helper()
	for _, b := range []string{"git", "gofmt"} {
		if _, err := exec.LookPath(b); err != nil {
			t.Skipf("%s not on PATH", b)
		}
	}
	dir := t.TempDir()
	if r, err := filepath.EvalSymlinks(dir); err == nil {
		dir = r
	}
	git := func(args ...string) {
		t.Helper()
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	git("init", "-q")
	git("config", "user.email", "t@example.com")
	git("config", "user.name", "T")
	write := func(rel, body string) {
		if err := os.MkdirAll(filepath.Dir(filepath.Join(dir, rel)), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, rel), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/p\n\ngo 1.22\n")
	write("p.go", "package p\n\n// F does a thing.\n//\n// The story of 2026-09-12.\nfunc F() int { return 1 }\n")
	write("docs/a.md", "# a\n")
	write("README.md", "hello\n")
	git("add", "go.mod", "p.go", "docs/a.md", "README.md")
	git("commit", "-q", "-m", "base")
	wd, _ := os.Getwd()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(wd) })
	return dir
}

func coRun(t *testing.T, args ...string) (int, string) {
	t.Helper()
	var out bytes.Buffer
	code := runCommentsOnly(append(args, "-no-checks"), &out)
	return code, out.String()
}

func TestCommentsOnly_gitCommitGreenAndRed(t *testing.T) {
	dir := coRepo(t)
	commit := func(msg string, paths ...string) {
		t.Helper()
		for _, args := range [][]string{append([]string{"add"}, paths...), {"commit", "-q", "-m", msg}} {
			c := exec.Command("git", args...)
			c.Dir = dir
			if out, err := c.CombinedOutput(); err != nil {
				t.Fatalf("git %v: %v\n%s", args, err, out)
			}
		}
	}
	// a comments commit with the story moved verbatim into docs
	os.WriteFile(filepath.Join(dir, "p.go"), []byte("package p\n\n// F does a thing; history in docs/code-notes/p.md#F.\nfunc F() int { return 1 }\n"), 0o644)
	os.MkdirAll(filepath.Join(dir, "docs/code-notes"), 0o755)
	os.WriteFile(filepath.Join(dir, "docs/code-notes/p.md"), []byte("## F\n\nThe story of 2026-09-12.\n"), 0o644)
	commit("comments", "p.go", "docs/code-notes/p.md")
	code, out := coRun(t, "HEAD~1")
	if code != 0 || !strings.Contains(out, "GREEN") || !strings.Contains(out, "found verbatim") {
		t.Fatalf("comments commit: want green with the story found, got %d\n%s", code, out)
	}
	// the same base, a code change hiding in a later commit
	os.WriteFile(filepath.Join(dir, "p.go"), []byte("package p\n\n// F does a thing; history in docs/code-notes/p.md#F.\nfunc F() int { return 2 }\n"), 0o644)
	commit("sneaky", "p.go")
	if code, out := coRun(t, "HEAD~2"); code != 1 || !strings.Contains(out, "CODE CHANGED") {
		t.Fatalf("code change: want red, got %d\n%s", code, out)
	}
	// a non-doc, non-go file in the diff
	os.WriteFile(filepath.Join(dir, "README.md"), []byte("changed\n"), 0o644)
	commit("readme", "README.md")
	if code, out := coRun(t, "HEAD~1"); code != 1 || !strings.Contains(out, "neither a .go file nor under docs/") {
		t.Fatalf("non-doc file: want red, got %d\n%s", code, out)
	}
	// a new .go file
	os.WriteFile(filepath.Join(dir, "q.go"), []byte("package p\n"), 0o644)
	commit("new file", "q.go")
	if code, out := coRun(t, "HEAD~1"); code != 1 || !strings.Contains(out, "was added") {
		t.Fatalf("added .go file: want red, got %d\n%s", code, out)
	}
}

func TestCommentsOnly_worktreeMode(t *testing.T) {
	dir := coRepo(t)
	os.WriteFile(filepath.Join(dir, "p.go"), []byte("package p\n\nfunc F() int { return 1 }\n"), 0o644)
	os.MkdirAll(filepath.Join(dir, "docs"), 0o755)
	os.WriteFile(filepath.Join(dir, "docs/new-notes.md"), []byte("## F\n\nThe story of 2026-09-12.\n"), 0o644) // untracked
	code, out := coRun(t, "--worktree", "HEAD")
	if code != 0 || !strings.Contains(out, "GREEN") || !strings.Contains(out, "found verbatim") {
		t.Fatalf("worktree comments edit with an untracked notes file: want green, got %d\n%s", code, out)
	}
	os.WriteFile(filepath.Join(dir, "p.go"), []byte("package p\n\nfunc F() int { return 9 }\n"), 0o644)
	if code, out := coRun(t, "--worktree", "HEAD"); code != 1 || !strings.Contains(out, "CODE CHANGED") {
		t.Fatalf("worktree code edit: want red, got %d\n%s", code, out)
	}
}

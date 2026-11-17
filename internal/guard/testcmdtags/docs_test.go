package testcmdtags

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// docCommand is a `go test` command a doc spells out, where it is.
type docCommand struct{ where, line string }

var (
	inlineCode = regexp.MustCompile("`([^`]*go test [^`]*)`")
	fenceOpen  = regexp.MustCompile("^\\s*```")
)

// notesCheckout is the private notes checkout beside root (../goinfer-notes), or "" when there is none, which is CI. The
// task, queue, plan and scoping docs live there once they leave the public tree.
func notesCheckout(root string) string {
	n := filepath.Join(root, "..", "goinfer-notes")
	if st, err := os.Stat(n); err == nil && st.IsDir() {
		return n
	}
	return ""
}

// docCommands lists the `go test` commands in the live docs under root/docs and, when the notes checkout is present, in its
// documents (named "notes:<path>"): inline code spans and fenced code lines. The dated records (completed, archive,
// measurements) are left alone: they say what was run then, at that day's paths.
func docCommands(t *testing.T, root string) []docCommand {
	t.Helper()
	out := walkDocCommands(t, filepath.Join(root, "docs"), root, "")
	if n := notesCheckout(root); n != "" {
		out = append(out, walkDocCommands(t, n, n, "notes:")...)
	}
	return out
}

func walkDocCommands(t *testing.T, dir, base, label string) []docCommand {
	t.Helper()
	var out []docCommand
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (d.Name() == "completed" || d.Name() == "archive" || d.Name() == "measurements" || d.Name() == ".git") {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(p, ".md") {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(base, p)
		fenced := false
		for i, line := range strings.Split(string(b), "\n") {
			at := label + filepath.ToSlash(rel) + ":" + strconv.Itoa(i+1)
			if fenceOpen.MatchString(line) {
				fenced = !fenced
				continue
			}
			if fenced {
				if strings.Contains(line, "go test ") {
					out = append(out, docCommand{at, strings.TrimSpace(line)})
				}
				continue
			}
			for _, m := range inlineCode.FindAllStringSubmatch(line, -1) {
				out = append(out, docCommand{at, m[1]})
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// TestDocCommands_readsTheNotesCheckout: a notes checkout beside the repository is read like docs/, minus its archive; without
// one only docs/ is read.
func TestDocCommands_readsTheNotesCheckout(t *testing.T) {
	base := t.TempDir()
	write := func(rel, text string) {
		p := filepath.Join(base, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	root := filepath.Join(base, "goinfer")
	write("goinfer/docs/a.md", "run `go test ./x -run TestA`\n")
	write("goinfer-notes/tasks/t.md", "```\ngo test ./y -run TestB\n```\n")
	write("goinfer-notes/completed/old.md", "run `go test ./z -run TestC`\n")
	write("goinfer-notes/measurements/m.md", "run `go test ./w -run TestD`\n")

	got := map[string]string{}
	for _, dc := range docCommands(t, root) {
		got[dc.where] = dc.line
	}
	want := map[string]string{
		"docs/a.md:1":        "go test ./x -run TestA",
		"notes:tasks/t.md:2": "go test ./y -run TestB",
	}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for k, v := range want {
		if got[k] != v {
			t.Fatalf("got %v, want %v", got, want)
		}
	}

	if err := os.RemoveAll(filepath.Join(base, "goinfer-notes")); err != nil {
		t.Fatal(err)
	}
	if n := len(docCommands(t, root)); n != 1 {
		t.Fatalf("without a notes checkout got %d commands, want the 1 under docs/", n)
	}
}

// TestDocs_goTestCommandsCompileTheTestsTheyName: every `go test ... -run` command in the live docs names tests its tags
// compile, in packages that exist. A command with a placeholder (<...>, ${...}) is not checked. It first proves the check fires.
func TestDocs_goTestCommandsCompileTheTestsTheyName(t *testing.T) {
	root := filepath.Join("..", "..", "..")
	planted := "go test -count=1 ./backend/metal/ -run 'TestFitQuery_|TestS18ShrinkCtxToFit' -v"
	c, _ := Parse(planted)
	if f, err := Check(root, c); err != nil || len(f) == 0 {
		t.Fatalf("VOID: an untagged run of the darwin && goinfer_testhooks fit tests is not flagged (%v, %v)", f, err)
	}
	checked := 0
	for _, dc := range docCommands(t, root) {
		if strings.Contains(dc.line, "<") || strings.Contains(dc.line, "${") {
			continue
		}
		c, ok := Parse(dc.line)
		if !ok {
			continue
		}
		f, err := Check(root, c)
		if err != nil {
			t.Errorf("%s: %v", dc.where, err)
			continue
		}
		checked++
		for _, x := range f {
			t.Errorf("%s: %s", dc.where, x)
		}
	}
	if checked == 0 {
		t.Fatal("VOID: no go test command found in the live docs")
	}
	if notesCheckout(root) == "" {
		t.Log("no notes checkout beside the repository: the task, queue, plan and scoping docs kept there were not read")
	}
	t.Logf("%d commands checked", checked)
}

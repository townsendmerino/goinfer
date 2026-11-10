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

// docCommands lists the `go test` commands in the live docs under root/docs: inline code spans and fenced code lines. The
// dated records (docs/completed, docs/archive, docs/measurements) are left alone: they say what was run then, at that day's paths.
func docCommands(t *testing.T, root string) []docCommand {
	t.Helper()
	var out []docCommand
	err := filepath.WalkDir(filepath.Join(root, "docs"), func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (d.Name() == "completed" || d.Name() == "archive" || d.Name() == "measurements") {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(p, ".md") {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, p)
		fenced := false
		for i, line := range strings.Split(string(b), "\n") {
			at := filepath.ToSlash(rel) + ":" + strconv.Itoa(i+1)
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
	t.Logf("%d commands checked", checked)
}

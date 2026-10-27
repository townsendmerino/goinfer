// Package movedpaths holds the test that keeps the aikit paths Track AX moved into goinfer out of goinfer's imports and
// module files (docs/tasks/task-subsystems-2026-10.md §7.11). It has no non-test code.
package movedpaths

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

const aikit = "github.com/townsendmerino/aikit/"

// moved reports whether an import path is one Track AX moved: aikit's vision and audio packages, and its gpu module
// with everything under it. The goinfer copies are internal/vision, internal/audio and the backends' internal/gpudev.
func moved(path string) bool {
	rest, ok := strings.CutPrefix(path, aikit)
	return ok && (rest == "vision" || rest == "audio" || rest == "gpu" || strings.HasPrefix(rest, "gpu/"))
}

// movedModule matches a go.mod or go.sum line naming aikit's gpu module or one of its sub-modules.
var movedModule = regexp.MustCompile(`github\.com/townsendmerino/aikit/gpu(/[^\s@]+)?[\s@]`)

func TestNoMovedAikitPath(t *testing.T) {
	var goFiles, modFiles int
	err := filepath.WalkDir("../..", func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", ".claude", "docs", "testdata", "node_modules":
				return filepath.SkipDir
			}
			return nil
		}
		switch name := d.Name(); {
		case strings.HasSuffix(name, ".go"):
			goFiles++
			f, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
			if err != nil {
				return err
			}
			for _, im := range f.Imports {
				if p, _ := strconv.Unquote(im.Path.Value); moved(p) {
					t.Errorf("%s imports %s: use goinfer's copy (internal/vision, internal/audio, or the backend's internal/gpudev)", path, p)
				}
			}
		case name == "go.mod" || name == "go.sum":
			modFiles++
			b, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			for i, line := range strings.Split(string(b), "\n") {
				if movedModule.MatchString(line + " ") {
					t.Errorf("%s:%d names aikit's gpu module, which goinfer no longer requires: %s", path, i+1, strings.TrimSpace(line))
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if goFiles < 1000 || modFiles < 10 {
		t.Fatalf("scanned %d .go files and %d go.mod/go.sum files: the walk read too little to mean anything", goFiles, modFiles)
	}
	t.Logf("scanned %d .go files and %d module files", goFiles, modFiles)
}

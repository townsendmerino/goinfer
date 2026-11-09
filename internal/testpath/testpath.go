// Package testpath finds the root module's files from a test, whatever directory the test's package is in: a path
// counted in "../" from the package goes stale when the package moves, and a test whose fixture path is wrong skips
// instead of failing.
package testpath

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const rootModule = "module github.com/townsendmerino/goinfer"

// Root is the root module's directory: the nearest directory at or above the working directory (the package's, under
// go test) whose go.mod declares goinfer's root module. It fails t when there is none.
func Root(t testing.TB) string {
	t.Helper()
	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatal(err)
	}
	for {
		if f, err := os.Open(filepath.Join(dir, "go.mod")); err == nil {
			sc := bufio.NewScanner(f)
			first := sc.Scan() && strings.TrimSpace(sc.Text()) == rootModule
			f.Close()
			if first {
				return dir
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("testpath: goinfer's root go.mod not found at or above the package directory")
		}
		dir = parent
	}
}

// Testdata is the path of elem under the root module's testdata directory. It does not check that the file exists: a
// test that skips on an absent asset still does.
func Testdata(t testing.TB, elem ...string) string {
	t.Helper()
	return filepath.Join(append([]string{Root(t), "testdata"}, elem...)...)
}

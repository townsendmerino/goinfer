package wgsl

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestNoBuiltinSinCos refuses a WGSL builtin sin or cos call in any shader here: rotations go through RopeCosSinWGSL.
func TestNoBuiltinSinCos(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	builtin := regexp.MustCompile(`(^|[^\w.])(sin|cos)\s*\(`)
	rawString := regexp.MustCompile("`[^`]*`")
	n := 0
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, lit := range rawString.FindAllString(string(src), -1) {
			for line := range strings.SplitSeq(lit, "\n") {
				if builtin.MatchString(line) {
					t.Errorf("%s: a shader calls a WGSL builtin sin or cos; use RopeCosSinWGSL: %s", f, strings.TrimSpace(line))
				}
			}
		}
		n++
	}
	if n == 0 {
		t.Fatal("no wgsl sources read")
	}
}

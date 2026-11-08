//go:build gpu

package kernels

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestRopeKernels_noBuiltinSinCos refuses a WGSL builtin sin or cos call in this package's shaders (its Go raw strings): a rotation
// goes through ropeCosSin, whose accuracy does not depend on the adapter.
func TestRopeKernels_noBuiltinSinCos(t *testing.T) {
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
		// Shader source is Go raw strings; scan only those, so a Go comment that mentions sin or cos is not a shader call.
		for _, lit := range rawString.FindAllString(string(src), -1) {
			for _, line := range strings.Split(lit, "\n") {
				if builtin.MatchString(line) {
					t.Errorf("%s: a shader calls a WGSL builtin sin or cos; use ropeCosSin: %s", f, strings.TrimSpace(line))
				}
			}
		}
		n++
	}
	if n == 0 {
		t.Fatal("no kernel sources read")
	}
}

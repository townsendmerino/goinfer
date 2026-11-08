//go:build gpu

package kernels

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestRopeKernels_noBuiltinSinCos refuses a WGSL builtin sin( or cos( in this package's kernels: a rotation goes through ropeCosSin,
// whose accuracy does not depend on the adapter. Go code calls math.Sin and math.Cos, never a lower-case sin( or cos(, so every match
// is shader source.
func TestRopeKernels_noBuiltinSinCos(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	builtin := regexp.MustCompile(`(^|[^\w.])(sin|cos)\(`)
	n := 0
	for _, f := range files {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(src), "\n") {
			if strings.HasPrefix(strings.TrimSpace(line), "//") {
				continue
			}
			if builtin.MatchString(line) {
				t.Errorf("%s:%d calls a WGSL builtin sin or cos; use ropeCosSin: %s", f, i+1, strings.TrimSpace(line))
			}
		}
		n++
	}
	if n == 0 {
		t.Fatal("no kernel sources read")
	}
}

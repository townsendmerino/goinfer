package gpudev

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

// R-11 (goinfer docs/tasks/task-recompute-audit.md §5): the two backends disagree on what a fresh buffer holds — CUDA's cuMemAlloc is uninitialized, Metal's
// newBufferWithLength is zero-filled — and both NewBufferLen* docs used to say "uninitialized". The behaviour is deliberately unchanged; the docs are the fix, and
// this pins them. It reads cuda.go and metal.go as source (each is build-tagged to its own OS), so it runs on either host and checks both files.
func gpuDocs(t *testing.T, file string) map[string]string {
	t.Helper()
	f, err := parser.ParseFile(token.NewFileSet(), file, nil, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	m := map[string]string{}
	for _, d := range f.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Doc != nil {
			m[fd.Name.Name] = strings.Join(strings.Fields(fd.Doc.Text()), " ") // one line: a phrase may wrap
		}
	}
	return m
}

func TestOutputContract_newBufferDocsAgreeWithBackends(t *testing.T) {
	cuda, metal := gpuDocs(t, "cuda.go"), gpuDocs(t, "metal.go")
	for _, fn := range []string{"NewBufferLen", "NewBufferBytes", "NewBufferLenOf"} {
		c, m := strings.ToLower(cuda[fn]), strings.ToLower(metal[fn])
		if c == "" || m == "" {
			t.Fatalf("%s: missing a doc comment (cuda %t, metal %t)", fn, c != "", m != "")
		}
		if !strings.Contains(c, "uninitialized") {
			t.Errorf("cuda.go %s: the doc must say the memory is uninitialized (cuMemAlloc does not clear it)", fn)
		}
		if !strings.Contains(m, "zero-filled") {
			t.Errorf("metal.go %s: the doc must say the memory is zero-filled (newBufferWithLength)", fn)
		}
		if strings.Contains(m, "uninitialized") && !strings.Contains(m, "cuda") {
			t.Errorf("metal.go %s: still calls Metal's buffer uninitialized", fn)
		}
		for name, d := range map[string]string{"cuda.go": c, "metal.go": m} {
			if !strings.Contains(d, "must not rely") {
				t.Errorf("%s %s: the doc must say that code shared across backends must not rely on zeros", name, fn)
			}
		}
	}
}

func TestOutputContract_hostWritersStateTheirs(t *testing.T) {
	cuda, metal := gpuDocs(t, "cuda.go"), gpuDocs(t, "metal.go")
	for _, c := range []struct {
		docs map[string]string
		file string
		fn   string
	}{{cuda, "cuda.go", "Download"}, {metal, "metal.go", "Download"}, {cuda, "cuda.go", "ReadToHost"}} {
		if !strings.Contains(c.docs[c.fn], "Output contract: overwrites") {
			t.Errorf("%s %s: no 'Output contract: overwrites …' line", c.file, c.fn)
		}
	}
	for _, file := range []string{"cuda_copy.go", "metal_copy.go"} {
		if !strings.Contains(gpuDocs(t, file)["CopyDevice"], "Output contract: overwrites") {
			t.Errorf("%s CopyDevice: no 'Output contract: overwrites …' line", file)
		}
	}
}

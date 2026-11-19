package mem

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// recoverExempt names the recovers that absorb every panic on purpose, as "path:function", each with its reason. A recover outside
// this list must let a programming bug through: classify the value with DeviceAllocFailed (which a runtime.Error never passes) or
// call RepanicBug first.
var recoverExempt = map[string]string{
	"internal/app/serveapp/panicguard.go:*":                    "the server's goroutine guards: a panic fails its one request, job or batch line, with the stack on stderr",
	"internal/app/serveapp/webui.go:handleWebLoad":             "the web-UI load goroutine's guard: the panic fails that load, with the stack on stderr",
	"internal/app/serveapp/vision_serve.go:recoverDeviceTower": "a tower panic fails its one request with the panic's value; nothing falls back",
	"internal/app/inproc/transport.go:RoundTrip":               "the in-process transport's handler boundary, as net/http's own: the panic fails that response",
	"decoder/gguf.go:ggufConfig":                               "hostile GGUF metadata drives a family builder into a divide by zero or an index panic: that is the malformed-input error",
	"decoder/mc3_batch.go:runSolo":                             "a panic fails that token's generation with its value, not the others' in the run; nothing falls back",
	"decoder/cpu_batch.go:runStep":                             "a panic fails that batched step's tokens with its value; nothing falls back",
	"backend/cuda/resident.go:pipeName":                        "an error path's name lookup must not panic: a failure reads as \"?\"",
	"backend/metal/workingset.go:deviceWorkingSet":             "a probe: a missing Metal symbol reads as an unknown working set (0)",
}

// classifies reports whether fn, the function whose body calls recover, lets a bug through: it calls DeviceAllocFailed or
// RepanicBug, or names runtime.Error (a type switch or assertion that treats it apart).
func classifies(fn ast.Node) bool {
	found := false
	ast.Inspect(fn, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.SelectorExpr:
			if x.Sel.Name == "DeviceAllocFailed" || x.Sel.Name == "RepanicBug" {
				found = true
			}
			if id, ok := x.X.(*ast.Ident); ok && id.Name == "runtime" && x.Sel.Name == "Error" {
				found = true
			}
		case *ast.Ident:
			if x.Name == "DeviceAllocFailed" || x.Name == "RepanicBug" {
				found = true
			}
		}
		return !found
	})
	return found
}

// recoverSites returns, for each recover() call in f, the name of the top-level declaration it is in and whether the innermost
// function holding it classifies the value.
func recoverSites(f *ast.File) (sites []struct {
	decl string
	ok   bool
	pos  token.Pos
}) {
	for _, d := range f.Decls {
		var name string
		switch x := d.(type) {
		case *ast.FuncDecl:
			name = x.Name.Name
		case *ast.GenDecl:
			for _, sp := range x.Specs {
				if vs, ok := sp.(*ast.ValueSpec); ok && len(vs.Names) > 0 {
					name = vs.Names[0].Name
				}
			}
		}
		var stack []ast.Node
		ast.Inspect(d, func(n ast.Node) bool {
			if n == nil {
				stack = stack[:len(stack)-1]
				return true
			}
			stack = append(stack, n)
			c, ok := n.(*ast.CallExpr)
			if !ok || len(c.Args) != 0 {
				return true
			}
			if id, ok := c.Fun.(*ast.Ident); !ok || id.Name != "recover" {
				return true
			}
			var fn ast.Node
			for i := len(stack) - 1; i >= 0 && fn == nil; i-- {
				switch stack[i].(type) {
				case *ast.FuncLit, *ast.FuncDecl:
					fn = stack[i]
				}
			}
			sites = append(sites, struct {
				decl string
				ok   bool
				pos  token.Pos
			}{name, fn != nil && classifies(fn), c.Pos()})
			return true
		})
	}
	return sites
}

// TestRecover_everyBroadRecoverLetsABugThrough: every non-test recover() in the repository, every module, either classifies the
// value (DeviceAllocFailed, RepanicBug, or a runtime.Error case) or is in recoverExempt; and no non-test string literal outside
// backend/cuda's gpudev spells the allocation failure's text, which only its AllocFailure type produces.
func TestRecover_everyBroadRecoverLetsABugThrough(t *testing.T) {
	plant := "package x\nfunc a() { defer func() { if p := recover(); p != nil { _ = p } }() }\n" +
		"func b() { defer func() { if p := recover(); p != nil { mem.RepanicBug(p) } }() }\n"
	pf, err := parser.ParseFile(token.NewFileSet(), "plant.go", plant, 0)
	if err != nil {
		t.Fatal(err)
	}
	if s := recoverSites(pf); len(s) != 2 || s[0].ok || !s[1].ok {
		t.Fatalf("the planted recovers read as %+v, want a unclassified and b classified", s)
	}

	root, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Fatalf("no repository root at %s", root)
	}
	const gpudevFile = "backend/cuda/internal/gpudev/cuda.go"
	var bad []string
	seen := map[string]bool{}
	read := 0
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			n := d.Name()
			if path != root && (n == "testdata" || n == "node_modules" || strings.HasPrefix(n, ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return err
		}
		read++
		rel, _ := filepath.Rel(root, path)
		rel = filepath.ToSlash(rel)
		for _, s := range recoverSites(f) {
			key := rel + ":" + s.decl
			if _, ok := recoverExempt[rel+":*"]; ok {
				seen[rel+":*"] = true
				continue
			}
			if _, ok := recoverExempt[key]; ok {
				seen[key] = true
				continue
			}
			if !s.ok {
				bad = append(bad, rel+": "+s.decl+" (line "+strconv.Itoa(fset.Position(s.pos).Line)+"): a recover that absorbs every panic, a bug's included: call RepanicBug(p) first, classify with DeviceAllocFailed, or add it to recoverExempt with its reason")
			}
		}
		if rel != gpudevFile {
			ast.Inspect(f, func(n ast.Node) bool {
				if lit, ok := n.(*ast.BasicLit); ok && lit.Kind == token.STRING && strings.Contains(lit.Value, deviceAllocFailedText+" (") {
					bad = append(bad, rel+": a string literal spells the allocation failure's text; panic with gpudev.AllocFailedPanic or wrap ErrDeviceAlloc")
				}
				return true
			})
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if read < 500 {
		t.Fatalf("read %d non-test files: a scan that reads little proves nothing", read)
	}
	sort.Strings(bad)
	for _, b := range bad {
		t.Error(b)
	}
	for k := range recoverExempt {
		if !seen[k] {
			t.Errorf("recoverExempt names %s, which holds no recover any more: remove the entry", k)
		}
	}
}

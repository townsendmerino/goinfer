package prezerolint

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// Audit R-11 (docs/tasks/task-recompute-audit.md §5): no aikit CPU kernel accumulates into its output, so a caller that
// zeroes a buffer and then hands it to a kernel whose documented contract is "overwrites dst" or "zeroes dst internally"
// is doing dead work (or hiding a missing write). aikit v1.56.1 put an `Output contract:` line on every exported writer;
// this test reads those lines from the PINNED aikit source and fails on a `clear(x)` (or `for i := range x { x[i] = 0 }`)
// that sits within a few statements of such a call taking x.
//
// What it does not do. It is a name-and-proximity check: it does not follow x through a helper, does not see a zeroed
// `make`, and matches the callee by NAME, so a goinfer function that happens to share a writer's name could trip it (the
// allowlist below is for that, with a reason). The GPU side's dead zeroing (zero Go slices uploaded into device buffers)
// is invisible to it. A clear before an ACCUMULATING write is correct and is not flagged, because those kernels are
// goinfer's own (attendQuery's ctx, moeMLP's out) and carry no aikit contract line.
var contractLine = regexp.MustCompile(`Output contract: (overwrites|zeroes)`)

// allowlist: "file:line" -> why a flagged zeroing is nonetheless needed. Empty is the goal.
var allowlist = map[string]string{}

// lookAhead is how many following statements in the same block may hold the call.
const lookAhead = 6

func writerNames(t *testing.T, aikitDir string) map[string]string {
	t.Helper()
	names := map[string]string{}
	for _, sub := range []string{"linalg", "embed", "encoder"} {
		paths, _ := filepath.Glob(filepath.Join(aikitDir, sub, "*.go"))
		for _, p := range paths {
			if strings.HasSuffix(p, "_test.go") {
				continue
			}
			f, err := parser.ParseFile(token.NewFileSet(), p, nil, parser.ParseComments)
			if err != nil {
				t.Fatalf("parse %s: %v", p, err)
			}
			for _, d := range f.Decls {
				fd, ok := d.(*ast.FuncDecl)
				if !ok || fd.Doc == nil || !fd.Name.IsExported() {
					continue
				}
				if k := contractLine.FindString(strings.Join(strings.Fields(fd.Doc.Text()), " ")); k != "" {
					names[fd.Name.Name] = k
				}
			}
		}
	}
	return names
}

func text(fset *token.FileSet, n ast.Node) string {
	var b bytes.Buffer
	_ = printer.Fprint(&b, fset, n)
	return b.String()
}

// target is the expression a call argument or a zeroing statement names: x for x, x[a:b] and a.b.
func target(fset *token.FileSet, e ast.Expr) string {
	if s, ok := e.(*ast.SliceExpr); ok {
		return text(fset, s.X)
	}
	return text(fset, e)
}

func zeroed(fset *token.FileSet, s ast.Stmt) string {
	switch s := s.(type) {
	case *ast.ExprStmt:
		if c, ok := s.X.(*ast.CallExpr); ok {
			if id, ok := c.Fun.(*ast.Ident); ok && id.Name == "clear" && len(c.Args) == 1 {
				return target(fset, c.Args[0])
			}
		}
	case *ast.RangeStmt:
		if s.Body != nil && len(s.Body.List) == 1 {
			if as, ok := s.Body.List[0].(*ast.AssignStmt); ok && len(as.Lhs) == 1 && len(as.Rhs) == 1 {
				if lit, ok := as.Rhs[0].(*ast.BasicLit); ok && (lit.Value == "0" || lit.Value == "0.0") {
					if ix, ok := as.Lhs[0].(*ast.IndexExpr); ok && text(fset, ix.X) == text(fset, s.X) {
						return text(fset, s.X)
					}
				}
			}
		}
	}
	return ""
}

func callee(c *ast.CallExpr) string {
	switch f := c.Fun.(type) {
	case *ast.Ident:
		return f.Name
	case *ast.SelectorExpr:
		return f.Sel.Name
	}
	return ""
}

// findings scans one parsed file and returns "file:line: ..." for each zeroing followed by a contract writer taking it.
func findings(fset *token.FileSet, f *ast.File, writers map[string]string) []string {
	var out []string
	ast.Inspect(f, func(nd ast.Node) bool {
		var list []ast.Stmt
		switch b := nd.(type) {
		case *ast.BlockStmt:
			list = b.List
		case *ast.CaseClause:
			list = b.Body
		default:
			return true
		}
		for i, s := range list {
			tgt := zeroed(fset, s)
			if tgt == "" {
				continue
			}
			for j := i + 1; j < len(list) && j <= i+lookAhead; j++ {
				hit := ""
				ast.Inspect(list[j], func(x ast.Node) bool {
					c, ok := x.(*ast.CallExpr)
					if !ok || hit != "" {
						return true
					}
					if k, ok := writers[callee(c)]; ok {
						for _, a := range c.Args {
							if target(fset, a) == tgt {
								hit = callee(c) + " (" + k + ")"
							}
						}
					}
					return true
				})
				if hit != "" {
					pos := fset.Position(s.Pos())
					out = append(out, pos.Filename+":"+itoa(pos.Line)+": zeroes "+tgt+", then calls "+hit)
					break
				}
			}
		}
		return true
	})
	return out
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d []byte
	for ; n > 0; n /= 10 {
		d = append([]byte{byte('0' + n%10)}, d...)
	}
	return string(d)
}

func aikitDir(t *testing.T) string {
	t.Helper()
	out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", "github.com/townsendmerino/aikit").Output()
	if err != nil {
		t.Fatalf("go list -m aikit: %v (the module cache must hold the pinned aikit; run `go mod download`)", err)
	}
	return strings.TrimSpace(string(out))
}

// TestPreZero_scannerCanFire proves the scanner is not vacuous: planted pre-zeroing is found in both forms, and a clear before a
// call that is NOT a contract writer, or on a different buffer, is not.
func TestPreZero_scannerCanFire(t *testing.T) {
	writers := map[string]string{"MatmulBT": "Output contract: zeroes", "ExpF32Into": "Output contract: overwrites"}
	src := `package p
func a(x, dst, other []float32) {
	clear(dst)
	MatmulBT(x, x, dst[:4], 1, 2, 2)
}
func b(x, dst []float32) {
	for i := range dst {
		dst[i] = 0
	}
	ws.ExpF32Into(dst, x)
}
func c(x, dst, other []float32) {
	clear(dst)
	MatmulBT(x, x, other, 1, 2, 2) // different buffer
	clear(x)
	notAWriter(x)
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "planted.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	got := findings(fset, f, writers)
	if len(got) != 2 || !strings.Contains(got[0], "planted.go:3") || !strings.Contains(got[1], "planted.go:7") {
		t.Fatalf("scanner findings = %q; want exactly the two planted cases (lines 3 and 7)", got)
	}
}

func TestPreZero_noDeadZeroingBeforeOverwriteKernels(t *testing.T) {
	writers := writerNames(t, aikitDir(t))
	// If the pinned aikit predates the contract docs (v1.56.1), the set is empty or tiny and the check would pass vacuously.
	for _, must := range []string{"MatmulBT", "MatmulBTW4A8Into", "SoftmaxRowInto", "ExpF32Into", "QuantizeActivationsInto", "DequantizeRowInt4"} {
		if writers[must] == "" {
			t.Fatalf("pinned aikit has no `Output contract:` line on %s (%d writers found): it predates v1.56.1, so this check would pass vacuously", must, len(writers))
		}
	}
	if len(writers) < 60 {
		t.Fatalf("only %d writers with an overwrite/zeroes contract found in the pinned aikit; expected at least 60", len(writers))
	}
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	var bad []string
	scanned := 0
	_ = filepath.Walk(root, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			switch info.Name() {
			case ".git", "testdata", "docs", "vendor", "node_modules":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, p, nil, 0)
		if err != nil {
			return nil // a file that does not parse is the compiler's to report
		}
		scanned++
		for _, fnd := range findings(fset, f, writers) {
			rel, _ := filepath.Rel(root, p)
			fnd = strings.Replace(fnd, p, rel, 1)
			key := fnd[:strings.Index(fnd, ": zeroes")]
			if why, ok := allowlist[key]; ok {
				t.Logf("allowed %s: %s", fnd, why)
				continue
			}
			bad = append(bad, fnd)
		}
		return nil
	})
	if scanned < 200 {
		t.Fatalf("scanned only %d Go files under %s; the walk is wrong, so a pass would mean nothing", scanned, root)
	}
	t.Logf("scanned %d files against %d overwrite/zeroes writers", scanned, len(writers))
	for _, b := range bad {
		t.Errorf("%s — the contract says the callee overwrites (or zeroes internally); the clear is dead work. Delete it, or add it to allowlist with the reason it is needed.", b)
	}
}

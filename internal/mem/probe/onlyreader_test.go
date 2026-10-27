package probe

import (
	"bufio"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// osMemoryLiterals are the string literals an OS memory reading names: a /proc file, a field of one, a sysctl, a Windows
// or Mach entry point. A literal matches when it equals one, or, for the entries ending in "_" or starting with "/proc/",
// when it starts with or contains one.
var osMemoryLiterals = []string{
	"/proc/meminfo", "/proc/self/status", "/proc/self/statm", "/proc/self/smaps",
	"MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:", "VmSwap", "VmSwap:", "VmHWM", "VmHWM:", "VmRSS", "VmRSS:",
	"statm", "smaps", "smaps_rollup",
	"hw.memsize", "vm.swapusage", "kern.memorystatus_level", "vm.page_",
	"GlobalMemoryStatusEx", "host_statistics64", "task_info", "mach_host_self",
}

// osMemorySelectors are the syscall-package entry points that read memory: pkg.Name with Name one of these.
var osMemorySelectors = map[string]bool{"Getrusage": true, "Sysinfo": true, "SYS_MINCORE": true, "Mincore": true}

func literalReadsOSMemory(s string) bool {
	for _, p := range osMemoryLiterals {
		switch {
		case s == p:
			return true
		case strings.HasSuffix(p, "_") && strings.HasPrefix(s, p):
			return true
		case strings.HasPrefix(p, "/proc/") && strings.Contains(s, p):
			return true
		}
	}
	return false
}

// osMemoryReads lists the OS memory reads in one Go source file: its string literals, its selectors and its
// //go:cgo_import_dynamic directives. Comments other than that directive do not count.
func osMemoryReads(fset *token.FileSet, f *ast.File) []string {
	var out []string
	ast.Inspect(f, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.BasicLit:
			if x.Kind == token.STRING {
				if s, err := strconv.Unquote(x.Value); err == nil && literalReadsOSMemory(s) {
					out = append(out, fset.Position(x.Pos()).String()+": "+x.Value)
				}
			}
		case *ast.SelectorExpr:
			if osMemorySelectors[x.Sel.Name] {
				out = append(out, fset.Position(x.Pos()).String()+": ."+x.Sel.Name)
			}
		}
		return true
	})
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			if rest, ok := strings.CutPrefix(c.Text, "//go:cgo_import_dynamic "); ok {
				for _, w := range strings.Fields(rest) {
					if literalReadsOSMemory(w) {
						out = append(out, fset.Position(c.Pos()).String()+": "+c.Text)
					}
				}
			}
		}
	}
	return out
}

// repoRoot is the root module's directory: the nearest parent holding goinfer's go.mod.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatal(err)
	}
	for {
		if b, err := os.ReadFile(filepath.Join(dir, "go.mod")); err == nil {
			sc := bufio.NewScanner(strings.NewReader(string(b)))
			if sc.Scan() && strings.TrimSpace(sc.Text()) == "module github.com/townsendmerino/goinfer" {
				return dir
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("goinfer's root go.mod not found above the package")
		}
		dir = parent
	}
}

// TestOnlyProbeReadsOSMemory (MM0 gate G-MM0-3): no non-test Go file in any of the repository's modules, outside this
// package and testdata, reads the operating system's account of memory. It first proves it can go red: a planted source
// for each pattern is flagged.
func TestOnlyProbeReadsOSMemory(t *testing.T) {
	for _, p := range osMemoryLiterals {
		src := "package x\nvar _ = " + strconv.Quote(p) + "\n"
		if strings.HasSuffix(p, "_") {
			src = "package x\nvar _ = " + strconv.Quote(p+"free_count") + "\n"
		}
		if f, err := parser.ParseFile(token.NewFileSet(), "plant.go", src, parser.ParseComments); err != nil || len(osMemoryReads(token.NewFileSet(), f)) == 0 {
			t.Errorf("a planted %q was not flagged (err %v)", p, err)
		}
	}
	for sel := range osMemorySelectors {
		src := "package x\nimport \"syscall\"\nvar _ = syscall." + sel + "\n"
		if f, err := parser.ParseFile(token.NewFileSet(), "plant.go", src, parser.ParseComments); err != nil || len(osMemoryReads(token.NewFileSet(), f)) == 0 {
			t.Errorf("a planted syscall.%s was not flagged (err %v)", sel, err)
		}
	}
	plant := "package x\n//go:cgo_import_dynamic libc_host_statistics64 host_statistics64 \"/usr/lib/libSystem.B.dylib\"\n"
	if f, _ := parser.ParseFile(token.NewFileSet(), "plant.go", plant, parser.ParseComments); len(osMemoryReads(token.NewFileSet(), f)) == 0 {
		t.Error("a planted cgo_import_dynamic of host_statistics64 was not flagged")
	}
	quiet := "package x\n// reads /proc/meminfo and hw.memsize\nvar _ = \"memory: VmHWM %d MB\"\n"
	if f, _ := parser.ParseFile(token.NewFileSet(), "plant.go", quiet, parser.ParseComments); len(osMemoryReads(token.NewFileSet(), f)) != 0 {
		t.Error("a comment or a print label was flagged as a read")
	}

	root := repoRoot(t)
	self, err := filepath.Abs(".")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var walked, parsed int
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() {
			if path != root && (name == "testdata" || strings.HasPrefix(name, ".") || name == "node_modules") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			return nil
		}
		walked++
		f, perr := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if perr != nil {
			t.Errorf("parse %s: %v", path, perr)
			return nil
		}
		parsed++
		if filepath.Dir(path) == self {
			return nil
		}
		for _, r := range osMemoryReads(fset, f) {
			rel, _ := filepath.Rel(root, r)
			t.Errorf("an OS memory read outside internal/mem/probe: %s", rel)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if parsed != walked || parsed < 500 {
		t.Fatalf("parsed %d of %d non-test Go files: a scan that reads little proves nothing", parsed, walked)
	}
	t.Logf("scanned %d non-test Go files", parsed)
}

// TestProbeImports (MM0 gate G-MM0-3): this package imports the standard library and golang.org/x/sys only.
func TestProbeImports(t *testing.T) {
	ents, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, e := range ents {
		if !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		f, err := parser.ParseFile(token.NewFileSet(), e.Name(), nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		n++
		for _, imp := range f.Imports {
			p, _ := strconv.Unquote(imp.Path.Value)
			first, _, _ := strings.Cut(p, "/")
			if strings.Contains(first, ".") && !strings.HasPrefix(p, "golang.org/x/sys/") {
				t.Errorf("%s imports %s", e.Name(), p)
			}
		}
	}
	if n == 0 {
		t.Fatal("no source files read")
	}
}

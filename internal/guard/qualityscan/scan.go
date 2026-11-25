// Package qualityscan holds the scans that keep a code-quality pattern from coming back once it has been fixed, and the
// ratchets that keep one from growing. They run nightly and by hand, never per push or in the day loop:
//
//	go test -tags quality_scan -v ./internal/guard/qualityscan/
//
// (.github/workflows/quality-scans.yml). Each scan is a test in a quality_scan-tagged file. A deliberate exception is a line
// in the scan's allowlist under testdata/, with its reason; a ratchet's line also carries the ceiling the site may not
// exceed, so the scan fails on growth and only reports shrinkage.
package qualityscan

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// File is one Go source file of the repository, parsed with comments.
type File struct {
	Path string // repository-relative, slash-separated
	Test bool   // a _test.go file
	Src  []byte
	AST  *ast.File
}

// Tree is every Go file git tracks or would track (untracked but not ignored), across all modules, outside testdata.
type Tree struct {
	Root  string
	Fset  *token.FileSet
	Files []*File
}

// FindRoot walks up from dir to the directory holding go.work.example, the repository root.
func FindRoot(dir string) (string, error) {
	dir, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.work.example")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", errors.New("qualityscan: no go.work.example above the working directory")
		}
		dir = parent
	}
}

// Load parses the tree under root. A file that does not parse is an error: a scan that skipped it would pass on code it
// never read.
func Load(root string) (*Tree, error) {
	out, err := exec.Command("git", "-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.go").Output()
	if err != nil {
		return nil, fmt.Errorf("qualityscan: git ls-files: %w", err)
	}
	t := &Tree{Root: root, Fset: token.NewFileSet()}
	for p := range strings.SplitSeq(string(out), "\x00") {
		if p == "" || skipped(p) {
			continue
		}
		src, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(p)))
		if errors.Is(err, os.ErrNotExist) {
			continue // deleted in the working tree, not yet in the index
		}
		if err != nil {
			return nil, err
		}
		f, err := parser.ParseFile(t.Fset, p, src, parser.ParseComments)
		if err != nil {
			return nil, fmt.Errorf("qualityscan: %w", err)
		}
		t.Files = append(t.Files, &File{Path: p, Test: strings.HasSuffix(p, "_test.go"), Src: src, AST: f})
	}
	return t, nil
}

func skipped(p string) bool {
	for seg := range strings.SplitSeq(p, "/") {
		if seg == "testdata" || seg == "node_modules" || strings.HasPrefix(seg, ".") {
			return true
		}
	}
	return false
}

// Line is the 1-based line of pos.
func (t *Tree) Line(pos token.Pos) int { return t.Fset.Position(pos).Line }

// FuncKey names a function declaration as path:Name, or path:Recv.Name for a method, the key allowlists use.
func FuncKey(path string, fd *ast.FuncDecl) string {
	if fd.Recv == nil || len(fd.Recv.List) == 0 {
		return path + ":" + fd.Name.Name
	}
	typ := fd.Recv.List[0].Type
	for {
		switch x := typ.(type) {
		case *ast.StarExpr:
			typ = x.X
			continue
		case *ast.IndexExpr:
			typ = x.X
			continue
		case *ast.IndexListExpr:
			typ = x.X
			continue
		}
		break
	}
	if id, ok := typ.(*ast.Ident); ok {
		return path + ":" + id.Name + "." + fd.Name.Name
	}
	return path + ":" + fd.Name.Name
}

// Entry is one allowlist line: a key, an optional ceiling, and the reason the site is allowed.
type Entry struct {
	Key     string
	Ceiling int // 0 when the line has none: the site is allowed at any count
	Reason  string
}

// ReadAllowlist reads a scan's allowlist. Each line is `<key> [<ceiling>] -- <reason>`; blank lines and lines starting with
// # are ignored. A line without a reason, a duplicate key, or a ceiling that is not a positive integer is an error.
func ReadAllowlist(path string) (map[string]Entry, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseAllowlist(raw)
}

// ParseAllowlist parses allowlist text; ReadAllowlist states the format.
func ParseAllowlist(raw []byte) (map[string]Entry, error) {
	out := map[string]Entry{}
	sc := bufio.NewScanner(bytes.NewReader(raw))
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		head, reason, ok := strings.Cut(line, " -- ")
		reason = strings.TrimSpace(reason)
		if !ok || reason == "" {
			return nil, fmt.Errorf("allowlist line %d: no ` -- <reason>`: %q", n, line)
		}
		fields := strings.Fields(head)
		e := Entry{Reason: reason}
		switch len(fields) {
		case 1:
			e.Key = fields[0]
		case 2:
			c, err := strconv.Atoi(fields[1])
			if err != nil || c < 1 {
				return nil, fmt.Errorf("allowlist line %d: ceiling %q is not a positive integer", n, fields[1])
			}
			e.Key, e.Ceiling = fields[0], c
		default:
			return nil, fmt.Errorf("allowlist line %d: want `<key> [<ceiling>] -- <reason>`: %q", n, line)
		}
		if _, dup := out[e.Key]; dup {
			return nil, fmt.Errorf("allowlist line %d: duplicate key %q", n, e.Key)
		}
		out[e.Key] = e
	}
	return out, sc.Err()
}

// Compare checks found (site key to count) against an allowlist. grown lists each site that is not allowed or whose count
// is over its ceiling: the scan fails on these. shrunk lists each allowed site now absent or under its ceiling: the
// allowlist line can be tightened or removed, and the scan only reports it.
func Compare(found map[string]int, allow map[string]Entry) (grown, shrunk []string) {
	for k, n := range found {
		e, ok := allow[k]
		switch {
		case !ok:
			grown = append(grown, fmt.Sprintf("%s (%d)", k, n))
		case e.Ceiling > 0 && n > e.Ceiling:
			grown = append(grown, fmt.Sprintf("%s (%d, ceiling %d)", k, n, e.Ceiling))
		case e.Ceiling > 0 && n < e.Ceiling:
			shrunk = append(shrunk, fmt.Sprintf("%s (%d, ceiling %d)", k, n, e.Ceiling))
		}
	}
	for k := range allow {
		if _, ok := found[k]; !ok {
			shrunk = append(shrunk, k+" (gone)")
		}
	}
	slices.Sort(grown)
	slices.Sort(shrunk)
	return grown, shrunk
}

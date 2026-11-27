package gatehygiene

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
)

// RunCommentsOnly implements `gate comments-only`.
func RunCommentsOnly(argv []string, w io.Writer) int {
	fs := flag.NewFlagSet("gate comments-only", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	worktree := fs.Bool("worktree", false, "compare <base> with the working tree (uncommitted edits) instead of with HEAD")
	noChecks := fs.Bool("no-checks", false, "stop after the comments-only check; skip gofmt, go vet and the citation lint")
	verbose := fs.Bool("v", false, "list every moved-text run, not only those not found")
	logDir := fs.String("logdir", "", "write the run record here, for -replay")
	replay := fs.String("replay", "", "regrade a run record instead of reading the repository")
	var pos []string
	for len(argv) > 0 {
		if err := fs.Parse(argv); err != nil {
			return 2
		}
		argv = fs.Args()
		if len(argv) > 0 {
			pos = append(pos, argv[0])
			argv = argv[1:]
		}
	}
	if *replay != "" {
		return ReplayCommentsOnly(w, *replay)
	}
	if len(pos) != 1 {
		fmt.Fprintln(os.Stderr, "usage: gate comments-only <base> [--worktree] [-no-checks] [-v] [-logdir DIR] | gate comments-only -replay FILE")
		return 2
	}
	base := pos[0]
	root, err := gatecore.RepoRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate comments-only: %v\n", err)
		return 2
	}
	if err := os.Chdir(root); err != nil {
		fmt.Fprintf(os.Stderr, "gate comments-only: %v\n", err)
		return 2
	}
	changes, err := changedPaths(base, *worktree)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate comments-only: %v\n", err)
		return 2
	}
	mode := "base..HEAD"
	if *worktree {
		mode = "base..working tree"
	}
	var prelude strings.Builder
	pw := io.MultiWriter(w, &prelude)
	fmt.Fprintln(pw, "gate comments-only")
	gatecore.GatherProvenance([][2]string{{"base", base}, {"compares", mode}, {"changed", fmt.Sprintf("%d path(s)", len(changes))}}).Write(pw)

	in := CommentsOnlyInput{Changes: changes, Files: map[string]CommentsOnlyFile{}, NoChecks: *noChecks, Verbose: *verbose}
	for _, c := range changes {
		if strings.HasPrefix(c.Path, "docs/") || !strings.HasSuffix(c.Path, ".go") {
			continue
		}
		basePath := c.Path
		switch c.Status {
		case "M":
		case "A":
			old, moved := gatecore.PreMovePath(c.Path)
			if !moved {
				continue
			}
			if _, err := showAt(base, old); err != nil {
				continue
			}
			basePath = old
		default:
			continue
		}
		old, err1 := showAt(base, basePath)
		cur, err2 := currentContent(c.Path, *worktree)
		if err1 != nil || err2 != nil {
			in.Files[c.Path] = CommentsOnlyFile{Unreadable: fmt.Sprintf("%s: cannot read both versions (%v, %v)", c.Path, err1, err2)}
			continue
		}
		in.Files[c.Path] = CommentsOnlyFile{Verdict: CompareComments(c.Path, old, cur)}
	}
	added, err := addedDocsText(base, *worktree)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate comments-only: %v\n", err)
		return 2
	}
	in.Added = added
	if reds, goFiles := CommentsOnlyReds(in); len(reds) == 0 && !in.NoChecks {
		in.Checks = runCommentsOnlyChecks(goFiles)
	}
	code := GradeCommentsOnly(w, in)
	if *logDir != "" {
		if err := os.MkdirAll(*logDir, 0o755); err != nil {
			fmt.Fprintf(w, "gate: the run record was not written: %v\n", err)
		} else {
			gatecore.WriteRunRecord(w, *logDir, &gatecore.GateConfig{Name: "comments-only", Decision: "comments"}, nil, gatecore.Provenance{}, false,
				gatecore.RunRecordExtra{Prelude: prelude.String(), CommentsOnly: &in})
		}
	}
	return code
}

// ReplayCommentsOnly regrades a recorded comments-only run record.
func ReplayCommentsOnly(w io.Writer, path string) int {
	rec, err := gatecore.ReadRunRecord(path)
	if err != nil {
		fmt.Fprintf(w, "gate comments-only -replay: %v\n", err)
		return 2
	}
	in, err := gatecore.UnmarshalPayload[CommentsOnlyInput](rec.CommentsOnly)
	if err != nil {
		fmt.Fprintf(w, "gate comments-only -replay: %v\n", err)
		return 2
	}
	if in == nil {
		fmt.Fprintf(w, "gate comments-only -replay: %s is not a comments-only run record\n", path)
		return 2
	}
	fmt.Fprint(w, rec.Prelude)
	return GradeCommentsOnly(w, *in)
}

func changedPaths(base string, worktree bool) ([]PathChange, error) {
	args := []string{"diff", "--name-status", "--no-renames", "-z", base}
	if !worktree {
		args = append(args, "HEAD")
	}
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		return nil, fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	var cs []PathChange
	parts := strings.Split(strings.TrimRight(string(out), "\x00"), "\x00")
	for i := 0; i+1 < len(parts); i += 2 {
		cs = append(cs, PathChange{Path: parts[i+1], Status: parts[i]})
	}
	if worktree {
		un, err := exec.Command("git", "ls-files", "--others", "--exclude-standard", "-z").Output()
		if err != nil {
			return nil, fmt.Errorf("git ls-files --others: %w", err)
		}
		for p := range strings.SplitSeq(strings.TrimRight(string(un), "\x00"), "\x00") {
			if p != "" {
				cs = append(cs, PathChange{Path: p, Status: "A"})
			}
		}
	}
	return cs, nil
}

func showAt(rev, path string) ([]byte, error) {
	return exec.Command("git", "show", rev+":"+path).Output()
}

func currentContent(path string, worktree bool) ([]byte, error) {
	if worktree {
		return os.ReadFile(path)
	}
	return showAt("HEAD", path)
}

func addedDocsText(base string, worktree bool) (string, error) {
	args := []string{"diff", "-U0", "--no-renames", base}
	if !worktree {
		args = append(args, "HEAD")
	}
	args = append(args, "--", "docs")
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		return "", fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	var b strings.Builder
	for ln := range strings.SplitSeq(string(out), "\n") {
		if strings.HasPrefix(ln, "+") && !strings.HasPrefix(ln, "+++") {
			b.WriteString(ln[1:])
			b.WriteByte('\n')
		}
	}
	if worktree {
		un, err := exec.Command("git", "ls-files", "--others", "--exclude-standard", "-z", "--", "docs").Output()
		if err != nil {
			return "", err
		}
		for p := range strings.SplitSeq(strings.TrimRight(string(un), "\x00"), "\x00") {
			if p == "" {
				continue
			}
			if data, err := os.ReadFile(p); err == nil {
				b.Write(data)
				b.WriteByte('\n')
			}
		}
	}
	return b.String(), nil
}

// CompareComments checks that old and cur have identical non-comment token streams and protected comments.
func CompareComments(path string, old, cur []byte) FileVerdict {
	var v FileVerdict
	oldTok, err := tokensWithoutComments(old)
	if err != nil {
		v.Reds = append(v.Reds, fmt.Sprintf("%s: the base version does not scan: %v", path, err))
		return v
	}
	newTok, err := tokensWithoutComments(cur)
	if err != nil {
		v.Reds = append(v.Reds, fmt.Sprintf("%s: the new version does not scan: %v", path, err))
		return v
	}
	if i := firstTokenDifference(oldTok, newTok); i >= 0 {
		v.Reds = append(v.Reds, fmt.Sprintf("%s: CODE CHANGED — %s", path, describeTokenDifference(oldTok, newTok, i)))
	}
	of, ofs, err := parseWithComments(old)
	if err != nil {
		v.Reds = append(v.Reds, fmt.Sprintf("%s: the base version does not parse: %v", path, err))
		return v
	}
	nf, nfs, err := parseWithComments(cur)
	if err != nil {
		v.Reds = append(v.Reds, fmt.Sprintf("%s: the new version does not parse: %v", path, err))
		return v
	}
	oldProt, newProt := protectedComments(path, of, ofs), protectedComments(path, nf, nfs)
	if !slices.Equal(oldProt, newProt) {
		v.Reds = append(v.Reds, fmt.Sprintf("%s: PROTECTED COMMENT CHANGED — %s", path, describeProtectedDifference(oldProt, newProt)))
	}
	if line := directiveLookalike(nf, nfs); line != "" && directiveLookalike(of, ofs) == "" {
		v.Reds = append(v.Reds, fmt.Sprintf("%s: a comment line starts `//` + spaces + `go:` (%s); staticcheck SA9009 reads it as an ineffectual directive, so rewrap it", path, line))
	}
	v.CommentLinesOld, v.CommentLinesNew = commentLineCount(of, ofs), commentLineCount(nf, nfs)
	v.Removed = removedRuns(path, of, ofs, nf, nfs)
	return v
}

type tok struct {
	t   token.Token
	lit string
	ln  int
}

func (t tok) String() string {
	if t.lit != "" {
		return fmt.Sprintf("%s %q", t.t, t.lit)
	}
	return t.t.String()
}

func tokensWithoutComments(src []byte) ([]tok, error) {
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(src))
	var errs []string
	var s scanner.Scanner
	s.Init(file, src, func(pos token.Position, msg string) {
		errs = append(errs, fmt.Sprintf("%d:%d %s", pos.Line, pos.Column, msg))
	}, 0)
	var out []tok
	for {
		pos, t, lit := s.Scan()
		if t == token.EOF {
			break
		}
		out = append(out, tok{t, lit, fset.Position(pos).Line})
	}
	if len(errs) > 0 {
		return nil, fmt.Errorf("%s", errs[0])
	}
	return out, nil
}

func firstTokenDifference(a, b []tok) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i].t != b[i].t || a[i].lit != b[i].lit {
			return i
		}
	}
	if len(a) != len(b) {
		return min(len(a), len(b))
	}
	return -1
}

func describeTokenDifference(a, b []tok, i int) string {
	at := func(ts []tok) string {
		if i < len(ts) {
			s := ts[i].String()
			if len(s) > 60 {
				s = s[:60] + "…"
			}
			return fmt.Sprintf("%s (line %d)", s, ts[i].ln)
		}
		return "end of file"
	}
	return fmt.Sprintf("the first differing token (#%d) is %s before and %s after", i, at(a), at(b))
}

func parseWithComments(src []byte) (*ast.File, *token.FileSet, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	return f, fset, err
}

var protectedPrefixes = []string{"//go:", "// +build", "//+build", "//nolint", "//lint:ignore"}

func protectedComments(path string, f *ast.File, _ *token.FileSet) []string {
	var examples []*ast.FuncDecl
	if strings.HasSuffix(path, "_test.go") {
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil && strings.HasPrefix(fd.Name.Name, "Example") && fd.Body != nil {
				examples = append(examples, fd)
			}
		}
	}
	var out []string
	for _, g := range f.Comments {
		for _, c := range g.List {
			tag := ""
			for _, p := range protectedPrefixes {
				if strings.HasPrefix(c.Text, p) {
					tag = "directive"
				}
			}
			for _, fd := range examples {
				if c.Pos() >= fd.Body.Pos() && c.End() <= fd.Body.End() {
					tag = "in " + fd.Name.Name
				}
			}
			if tag != "" {
				out = append(out, tag+": "+c.Text)
			}
		}
	}
	return out
}

func directiveLookalike(f *ast.File, fset *token.FileSet) string {
	for _, g := range f.Comments {
		for _, c := range g.List {
			if !strings.HasPrefix(c.Text, "//") {
				continue
			}
			s := strings.TrimLeft(c.Text[2:], " \t")
			if len(s) != len(c.Text)-2 && strings.HasPrefix(s, "go:") {
				return fmt.Sprintf("line %d", fset.Position(c.Pos()).Line)
			}
		}
	}
	return ""
}

func describeProtectedDifference(a, b []string) string {
	for i := 0; i < len(a) || i < len(b); i++ {
		var x, y string
		if i < len(a) {
			x = a[i]
		}
		if i < len(b) {
			y = b[i]
		}
		if x != y {
			return fmt.Sprintf("#%d was %q, now %q", i, x, y)
		}
	}
	return "differs"
}

func commentLineCount(f *ast.File, fset *token.FileSet) int {
	n := 0
	for _, g := range f.Comments {
		for _, c := range g.List {
			n += fset.Position(c.End()).Line - fset.Position(c.Pos()).Line + 1
		}
	}
	return n
}

type cline struct {
	text string
	line int
}

func commentLines(g *ast.CommentGroup, fset *token.FileSet) []cline {
	var out []cline
	for _, c := range g.List {
		start := fset.Position(c.Pos()).Line
		body := c.Text
		switch {
		case strings.HasPrefix(body, "//"):
			body = body[2:]
		case strings.HasPrefix(body, "/*"):
			body = strings.TrimSuffix(strings.TrimPrefix(body, "/*"), "*/")
		}
		for i, ln := range strings.Split(body, "\n") {
			ln = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(ln), "*"))
			if ln != "" {
				out = append(out, cline{ln, start + i})
			}
		}
	}
	return out
}

func removedRuns(path string, of *ast.File, ofs *token.FileSet, nf *ast.File, nfs *token.FileSet) []RemovedRun {
	kept := map[string]bool{}
	for _, g := range nf.Comments {
		for _, l := range commentLines(g, nfs) {
			kept[l.text] = true
		}
	}
	var runs []RemovedRun
	for _, g := range of.Comments {
		var cur *RemovedRun
		var curLines []string
		flush := func() {
			if cur != nil {
				cur.Text = strings.Join(curLines, "\n")
				runs = append(runs, *cur)
				cur, curLines = nil, nil
			}
		}
		for _, l := range commentLines(g, ofs) {
			if kept[l.text] {
				flush()
				continue
			}
			if cur == nil {
				cur = &RemovedRun{File: path, Line: l.line}
			}
			cur.Lines++
			curLines = append(curLines, l.text)
		}
		flush()
	}
	return runs
}

func runCommentsOnlyChecks(goFiles []string) []CommentsOnlyCheck {
	var out []CommentsOnlyCheck
	step := func(kind, name string, cmd *exec.Cmd) {
		var buf bytes.Buffer
		cmd.Stdout, cmd.Stderr = &buf, &buf
		err := cmd.Run()
		out = append(out, CommentsOnlyCheck{Kind: kind, Name: name, OK: err == nil, Err: fmt.Sprint(err), Out: buf.String()})
	}
	if len(goFiles) > 0 {
		step("gofmt", "gofmt -l", exec.Command("gofmt", append([]string{"-l"}, goFiles...)...))
	}
	for _, pk := range touchedPackages(goFiles) {
		for _, tags := range pk.tagSets() {
			args := []string{"vet"}
			label := "go vet " + pk.dir
			if tags != "" {
				args = append(args, "-tags", tags)
				label += " -tags '" + tags + "'"
			}
			cmd := exec.Command("go", append(args, pk.pattern)...)
			cmd.Dir = pk.moduleDir
			cmd.Env = append(os.Environ(), pk.env...)
			step("step", label, cmd)
		}
	}
	step("step", "citation lint", exec.Command("python3", "scripts/queue_citation_lint.py"))
	return out
}

type vetTarget struct {
	moduleDir string
	dir       string
	pattern   string
	module    string
	env       []string
}

func (v vetTarget) tagSets() []string {
	switch v.module {
	case ".":
		return []string{"", "realckpt goinfer_testhooks"}
	case "backend/cuda", "cuda":
		return []string{"cuda goinfer_testhooks"}
	case "backend/webgpu", "gpu":
		return []string{"", "gpu goinfer_testhooks"}
	}
	return []string{""}
}

func touchedPackages(goFiles []string) []vetTarget {
	root, _ := os.Getwd()
	seen := map[string]bool{}
	var out []vetTarget
	for _, f := range goFiles {
		dir := filepath.Dir(f)
		mod := dir
		for mod != "." {
			if _, err := os.Stat(filepath.Join(root, mod, "go.mod")); err == nil {
				break
			}
			mod = filepath.Dir(mod)
		}
		rel, _ := filepath.Rel(mod, dir)
		pat := "./" + filepath.ToSlash(rel)
		if rel == "." {
			pat = "."
		}
		key := mod + "|" + pat
		if seen[key] {
			continue
		}
		seen[key] = true
		t := vetTarget{moduleDir: filepath.Join(root, mod), dir: dir, pattern: pat, module: mod}
		if mod == "backend/metal" || mod == "metal" {
			t.env = []string{"GOOS=darwin", "GOARCH=arm64", "CGO_ENABLED=0"}
		}
		out = append(out, t)
	}
	return out
}

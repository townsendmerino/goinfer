package main

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
)

// gate comments-only <base> [--worktree] proves that a commit (or the working tree) changed Go comments and docs and nothing else:
// the gate behind docs/tasks/task-code-comments-2026-10.md (CC2). A comment rewrite is only safe to review quickly if a tool, not
// the reviewer, vouches that no code moved with it.
//
// What is checked, per changed .go file:
//
//   - The token stream with comments skipped is identical before and after. String literals are tokens, so an edit inside an
//     embedded MSL, CUDA or WGSL kernel string is a code change and red. The task text says "print both with go/printer"; a token
//     stream is the same test without printer's blank-line rule, which keeps a blank line when two statements were two lines apart,
//     so deleting a comment line between two statements would print differently and read as a code change.
//   - Protected comments are byte-identical, in order: `//go:` directives, `// +build`, `//nolint`, `//lint:ignore`, and every
//     comment inside an Example function (its `// Output:` block is the test's expectation).
//
// What else is red: a changed path that is neither a .go file nor under docs/, and a .go file that was added, deleted or renamed
// (new code is a code change).
//
// What is listed and not red: the moved-text report. Each run of removed comment lines is searched for in the docs text the same
// diff adds, so "move, don't delete" is checkable; runs not found are the reviewer's to read against the doc the new comment cites.
//
// When the comments check is green the cheap gates for the touched modules follow: gofmt -l on the changed files, go vet per
// touched package (untagged, then with the tags that package family needs), and the citation lint. -no-checks stops after the
// comments check. Exit: 0 green, 1 red, 2 refused (bad arguments, no git).

func runCommentsOnly(argv []string, w io.Writer) int {
	fs := flag.NewFlagSet("gate comments-only", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	worktree := fs.Bool("worktree", false, "compare <base> with the working tree (uncommitted edits) instead of with HEAD")
	noChecks := fs.Bool("no-checks", false, "stop after the comments-only check; skip gofmt, go vet and the citation lint")
	verbose := fs.Bool("v", false, "list every moved-text run, not only those not found")
	// the flags may come before or after <base>
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
	if len(pos) != 1 {
		fmt.Fprintln(os.Stderr, "usage: gate comments-only <base> [--worktree] [-no-checks] [-v]")
		return 2
	}
	base := pos[0]
	root, err := repoRoot()
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
	fmt.Fprintln(w, "gate comments-only")
	gatherProvenance([][2]string{{"base", base}, {"compares", mode}, {"changed", fmt.Sprintf("%d path(s)", len(changes))}}).write(w)

	var reds []string
	var goFiles []string
	var removed []removedRun
	before, after := 0, 0
	for _, c := range changes {
		switch {
		case strings.HasPrefix(c.path, "docs/"):
		case strings.HasSuffix(c.path, ".go") && c.status == "M":
			goFiles = append(goFiles, c.path)
			old, err1 := showAt(base, c.path)
			cur, err2 := currentContent(c.path, *worktree)
			if err1 != nil || err2 != nil {
				reds = append(reds, fmt.Sprintf("%s: cannot read both versions (%v, %v)", c.path, err1, err2))
				continue
			}
			v := compareComments(c.path, old, cur)
			reds = append(reds, v.reds...)
			removed = append(removed, v.removed...)
			before += v.commentLinesOld
			after += v.commentLinesNew
		case strings.HasSuffix(c.path, ".go"):
			reds = append(reds, fmt.Sprintf("%s: a .go file was %s; a comments commit does not add, delete or rename code", c.path, changeKind(c.status)))
		default:
			reds = append(reds, fmt.Sprintf("%s: neither a .go file nor under docs/; only Go comments and docs may change in a comments commit", c.path))
		}
	}

	added, err := addedDocsText(base, *worktree)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate comments-only: %v\n", err)
		return 2
	}
	report := movedTextReport(removed, added)
	fmt.Fprintf(w, "\ncomment lines in the changed .go files: %d -> %d\n", before, after)
	report.write(w, *verbose)

	if len(reds) > 0 {
		fmt.Fprintf(w, "\nRED: %d finding(s) — this diff is NOT comments-only:\n", len(reds))
		for _, r := range reds {
			fmt.Fprintf(w, "  %s\n", r)
		}
		if !*noChecks {
			fmt.Fprintln(w, "\nnot run: gofmt, go vet and the citation lint (they describe a diff that is not the one this gate vouches for)")
		}
		return 1
	}
	fmt.Fprintf(w, "\nGREEN: %d .go file(s) differ only in comments; %d docs path(s) changed\n", len(goFiles), len(changes)-len(goFiles))
	if *noChecks {
		return 0
	}
	return runCommentsOnlyChecks(w, goFiles)
}

type pathChange struct{ path, status string }

func changeKind(s string) string {
	switch s {
	case "A":
		return "added"
	case "D":
		return "deleted"
	}
	return "renamed or copied (" + s + ")"
}

// changedPaths is every path that differs between base and HEAD (or the working tree), untracked files included in worktree mode.
func changedPaths(base string, worktree bool) ([]pathChange, error) {
	args := []string{"diff", "--name-status", "--no-renames", "-z", base}
	if !worktree {
		args = append(args, "HEAD")
	}
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		return nil, fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	var cs []pathChange
	parts := strings.Split(strings.TrimRight(string(out), "\x00"), "\x00")
	for i := 0; i+1 < len(parts); i += 2 {
		cs = append(cs, pathChange{path: parts[i+1], status: parts[i]})
	}
	if worktree {
		un, err := exec.Command("git", "ls-files", "--others", "--exclude-standard", "-z").Output()
		if err != nil {
			return nil, fmt.Errorf("git ls-files --others: %w", err)
		}
		for _, p := range strings.Split(strings.TrimRight(string(un), "\x00"), "\x00") {
			if p != "" {
				cs = append(cs, pathChange{path: p, status: "A"})
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

// addedDocsText is the text the diff adds under docs/ (a new untracked file counts in full in worktree mode).
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
	for _, ln := range strings.Split(string(out), "\n") {
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
		for _, p := range strings.Split(strings.TrimRight(string(un), "\x00"), "\x00") {
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

// ---- the per-file comparison (pure: no git, so the tests drive it directly) ----

type removedRun struct {
	file  string
	line  int    // first line of the run in the old file
	lines int    // comment lines in the run
	text  string // the run, markers stripped, one line per source line
}

type fileVerdict struct {
	reds                             []string
	removed                          []removedRun
	commentLinesOld, commentLinesNew int
}

func compareComments(path string, old, cur []byte) fileVerdict {
	var v fileVerdict
	oldTok, err := tokensWithoutComments(old)
	if err != nil {
		v.reds = append(v.reds, fmt.Sprintf("%s: the base version does not scan: %v", path, err))
		return v
	}
	newTok, err := tokensWithoutComments(cur)
	if err != nil {
		v.reds = append(v.reds, fmt.Sprintf("%s: the new version does not scan: %v", path, err))
		return v
	}
	if i := firstTokenDifference(oldTok, newTok); i >= 0 {
		v.reds = append(v.reds, fmt.Sprintf("%s: CODE CHANGED — %s", path, describeTokenDifference(oldTok, newTok, i)))
	}
	of, ofs, err := parseWithComments(old)
	if err != nil {
		v.reds = append(v.reds, fmt.Sprintf("%s: the base version does not parse: %v", path, err))
		return v
	}
	nf, nfs, err := parseWithComments(cur)
	if err != nil {
		v.reds = append(v.reds, fmt.Sprintf("%s: the new version does not parse: %v", path, err))
		return v
	}
	oldProt, newProt := protectedComments(path, of, ofs), protectedComments(path, nf, nfs)
	if !slices.Equal(oldProt, newProt) {
		v.reds = append(v.reds, fmt.Sprintf("%s: PROTECTED COMMENT CHANGED — %s", path, describeProtectedDifference(oldProt, newProt)))
	}
	v.commentLinesOld, v.commentLinesNew = commentLineCount(of, ofs), commentLineCount(nf, nfs)
	v.removed = removedRuns(path, of, ofs, nf, nfs)
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

// protectedComments lists, in source order, every comment this task never edits: directives, and anything inside an Example
// function of a test file. Each entry carries what a reviewer needs to find it.
func protectedComments(path string, f *ast.File, fset *token.FileSet) []string {
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

// commentLineCount is the number of source lines covered by comments (a trailing comment counts its own line).
func commentLineCount(f *ast.File, fset *token.FileSet) int {
	n := 0
	for _, g := range f.Comments {
		for _, c := range g.List {
			n += fset.Position(c.End()).Line - fset.Position(c.Pos()).Line + 1
		}
	}
	return n
}

// commentLines is a comment group as text lines, markers stripped and trimmed, blanks dropped, each with its source line.
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

// removedRuns: the comment lines the old file had that the new file's comments no longer contain, in runs of consecutive lines.
// A line that merely moved within the file is kept, not removed; a line is compared as text only.
func removedRuns(path string, of *ast.File, ofs *token.FileSet, nf *ast.File, nfs *token.FileSet) []removedRun {
	kept := map[string]bool{}
	for _, g := range nf.Comments {
		for _, l := range commentLines(g, nfs) {
			kept[l.text] = true
		}
	}
	var runs []removedRun
	for _, g := range of.Comments {
		var cur *removedRun
		var curLines []string
		flush := func() {
			if cur != nil {
				cur.text = strings.Join(curLines, "\n")
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
				cur = &removedRun{file: path, line: l.line}
			}
			cur.lines++
			curLines = append(curLines, l.text)
		}
		flush()
	}
	return runs
}

// ---- the moved-text report ----

type movedReport struct {
	found, missing []removedRun
}

func squash(s string) string { return strings.Join(strings.Fields(s), " ") }

// normalizeDocs strips the line prefixes a comment gains when it is pasted into markdown (`//`, `>`), then squashes whitespace, so
// "verbatim" survives being a blockquote or a code block.
func normalizeDocs(s string) string {
	var b strings.Builder
	for _, ln := range strings.Split(s, "\n") {
		ln = strings.TrimSpace(ln)
		ln = strings.TrimSpace(strings.TrimPrefix(ln, "//"))
		ln = strings.TrimSpace(strings.TrimPrefix(ln, ">"))
		b.WriteString(ln)
		b.WriteByte(' ')
	}
	return squash(b.String())
}

func movedTextReport(removed []removedRun, addedDocs string) movedReport {
	docs := normalizeDocs(addedDocs)
	var r movedReport
	for _, run := range removed {
		if strings.Contains(docs, squash(run.text)) {
			r.found = append(r.found, run)
		} else {
			r.missing = append(r.missing, run)
		}
	}
	return r
}

func (r movedReport) write(w io.Writer, verbose bool) {
	total := func(rs []removedRun) int {
		n := 0
		for _, x := range rs {
			n += x.lines
		}
		return n
	}
	fmt.Fprintf(w, "moved-text report: %d removed run(s), %d line(s): %d line(s) found verbatim in the docs this diff adds, %d line(s) NOT found (listed below, not red)\n",
		len(r.found)+len(r.missing), total(r.found)+total(r.missing), total(r.found), total(r.missing))
	show := func(label string, rs []removedRun) {
		for _, x := range rs {
			first := strings.SplitN(x.text, "\n", 2)[0]
			if len(first) > 90 {
				first = first[:90] + "…"
			}
			fmt.Fprintf(w, "  %-9s %s:%d (%d line(s)) %q\n", label, x.file, x.line, x.lines, first)
		}
	}
	if verbose {
		show("found", r.found)
	}
	show("NOT FOUND", r.missing)
}

// ---- the cheap gates that follow a green ----

func runCommentsOnlyChecks(w io.Writer, goFiles []string) int {
	red := false
	step := func(name string, cmd *exec.Cmd) {
		var buf bytes.Buffer
		cmd.Stdout, cmd.Stderr = &buf, &buf
		err := cmd.Run()
		if err != nil {
			red = true
			fmt.Fprintf(w, "RED   %s: %v\n%s", name, err, indent(buf.String()))
			return
		}
		fmt.Fprintf(w, "ok    %s\n", name)
	}
	fmt.Fprintln(w)
	if len(goFiles) > 0 {
		var out bytes.Buffer
		cmd := exec.Command("gofmt", append([]string{"-l"}, goFiles...)...)
		cmd.Stdout, cmd.Stderr = &out, &out
		if err := cmd.Run(); err != nil || out.Len() > 0 {
			red = true
			fmt.Fprintf(w, "RED   gofmt -l: %v\n%s", err, indent(out.String()))
		} else {
			fmt.Fprintf(w, "ok    gofmt -l (%d file(s))\n", len(goFiles))
		}
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
			step(label, cmd)
		}
	}
	step("citation lint", exec.Command("python3", "scripts/queue_citation_lint.py"))
	if red {
		fmt.Fprintln(w, "\nRED: the comments are comments-only, but a following check failed")
		return 1
	}
	fmt.Fprintln(w, "\nGREEN: comments-only, gofmt, go vet and the citation lint")
	return 0
}

func indent(s string) string {
	var b strings.Builder
	for _, ln := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
		b.WriteString("      " + ln + "\n")
	}
	return b.String()
}

type vetTarget struct {
	moduleDir string // absolute
	dir       string // the package directory, repo-relative, for labels
	pattern   string // as go vet takes it, relative to moduleDir
	module    string // repo-relative module directory, "." for the root
	env       []string
}

// tagSets: untagged always, then the tags CLAUDE.md names for that module's heavy code. metal is darwin-only and cgo-free, so it
// is vetted for darwin from here rather than skipped.
func (v vetTarget) tagSets() []string {
	switch v.module {
	case ".":
		return []string{"", "realckpt goinfer_testhooks"}
	case "cuda":
		return []string{"", "cuda goinfer_testhooks"}
	case "gpu":
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
		if mod == "metal" {
			t.env = []string{"GOOS=darwin", "GOARCH=arm64", "CGO_ENABLED=0"}
		}
		out = append(out, t)
	}
	return out
}

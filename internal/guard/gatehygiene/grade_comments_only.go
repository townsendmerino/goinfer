package gatehygiene

import (
	"fmt"
	"io"
	"strings"
)

// PathChange records one changed path and its git status letter.
type PathChange struct {
	Path   string `json:"path"`
	Status string `json:"status"`
}

// RemovedRun records a contiguous run of comment lines removed from a Go file.
type RemovedRun struct {
	File  string `json:"file"`
	Line  int    `json:"line"`
	Lines int    `json:"lines"`
	Text  string `json:"text"`
}

// FileVerdict holds CompareComments' findings and comment-line counts for one modified Go file.
type FileVerdict struct {
	Reds            []string     `json:"reds,omitempty"`
	Removed         []RemovedRun `json:"removed,omitempty"`
	CommentLinesOld int          `json:"comment_lines_old"`
	CommentLinesNew int          `json:"comment_lines_new"`
}

// CommentsOnlyFile records either a Go file's verdict or why its versions could not be read.
type CommentsOnlyFile struct {
	Verdict    FileVerdict `json:"verdict"`
	Unreadable string      `json:"unreadable,omitempty"`
}

// CommentsOnlyCheck records one post-check step (gofmt, vet, citation lint) run after a green comments check.
type CommentsOnlyCheck struct {
	Kind string `json:"kind"`
	Name string `json:"name"`
	OK   bool   `json:"ok"`
	Err  string `json:"err"`
	Out  string `json:"out,omitempty"`
}

// CommentsOnlyInput holds all recorded inputs graded by GradeCommentsOnly.
type CommentsOnlyInput struct {
	Changes  []PathChange                `json:"changes"`
	Files    map[string]CommentsOnlyFile `json:"files,omitempty"`
	Added    string                      `json:"added_docs"`
	NoChecks bool                        `json:"no_checks,omitempty"`
	Verbose  bool                        `json:"verbose,omitempty"`
	Checks   []CommentsOnlyCheck         `json:"checks,omitempty"`
}

func changeKind(s string) string {
	switch s {
	case "A":
		return "added"
	case "D":
		return "deleted"
	}
	return "renamed or copied (" + s + ")"
}

// CommentsOnlyReds returns red findings across changed paths and the modified .go file paths.
func CommentsOnlyReds(in CommentsOnlyInput) (reds, goFiles []string) {
	for _, c := range in.Changes {
		switch {
		case strings.HasPrefix(c.Path, "docs/"):
		case strings.HasSuffix(c.Path, ".go") && c.Status == "M":
			goFiles = append(goFiles, c.Path)
			f := in.Files[c.Path]
			if f.Unreadable != "" {
				reds = append(reds, f.Unreadable)
				continue
			}
			reds = append(reds, f.Verdict.Reds...)
		case strings.HasSuffix(c.Path, ".go"):
			reds = append(reds, fmt.Sprintf("%s: a .go file was %s; a comments commit does not add, delete or rename code", c.Path, changeKind(c.Status)))
		default:
			reds = append(reds, fmt.Sprintf("%s: neither a .go file nor under docs/; only Go comments and docs may change in a comments commit", c.Path))
		}
	}
	return reds, goFiles
}

// GradeCommentsOnly formats the comments-only verdict and optional post-check steps, returning the exit code.
func GradeCommentsOnly(w io.Writer, in CommentsOnlyInput) int {
	reds, goFiles := CommentsOnlyReds(in)
	var removed []RemovedRun
	before, after := 0, 0
	for _, c := range in.Changes {
		if f, ok := in.Files[c.Path]; ok && f.Unreadable == "" && strings.HasSuffix(c.Path, ".go") && c.Status == "M" && !strings.HasPrefix(c.Path, "docs/") {
			removed = append(removed, f.Verdict.Removed...)
			before += f.Verdict.CommentLinesOld
			after += f.Verdict.CommentLinesNew
		}
	}
	report := movedTextReport(removed, in.Added)
	fmt.Fprintf(w, "\ncomment lines in the changed .go files: %d -> %d\n", before, after)
	report.write(w, in.Verbose)

	if len(reds) > 0 {
		fmt.Fprintf(w, "\nRED: %d finding(s) — this diff is NOT comments-only:\n", len(reds))
		for _, r := range reds {
			fmt.Fprintf(w, "  %s\n", r)
		}
		if !in.NoChecks {
			fmt.Fprintln(w, "\nnot run: gofmt, go vet and the citation lint (they describe a diff that is not the one this gate vouches for)")
		}
		return 1
	}
	fmt.Fprintf(w, "\nGREEN: %d .go file(s) differ only in comments; %d docs path(s) changed\n", len(goFiles), len(in.Changes)-len(goFiles))
	if in.NoChecks {
		return 0
	}
	return GradeCommentsOnlyChecks(w, in.Checks, len(goFiles))
}

// GradeCommentsOnlyChecks formats post-check lines and returns the closing exit code.
func GradeCommentsOnlyChecks(w io.Writer, checks []CommentsOnlyCheck, nGo int) int {
	red := false
	fmt.Fprintln(w)
	for _, c := range checks {
		switch {
		case c.Kind == "gofmt" && (!c.OK || c.Out != ""):
			red = true
			fmt.Fprintf(w, "RED   gofmt -l: %s\n%s", c.Err, indent(c.Out))
		case c.Kind == "gofmt":
			fmt.Fprintf(w, "ok    gofmt -l (%d file(s))\n", nGo)
		case !c.OK:
			red = true
			fmt.Fprintf(w, "RED   %s: %s\n%s", c.Name, c.Err, indent(c.Out))
		default:
			fmt.Fprintf(w, "ok    %s\n", c.Name)
		}
	}
	if red {
		fmt.Fprintln(w, "\nRED: the comments are comments-only, but a following check failed")
		return 1
	}
	fmt.Fprintln(w, "\nGREEN: comments-only, gofmt, go vet and the citation lint")
	return 0
}

func indent(s string) string {
	var b strings.Builder
	for ln := range strings.SplitSeq(strings.TrimRight(s, "\n"), "\n") {
		b.WriteString("      " + ln + "\n")
	}
	return b.String()
}

type movedReport struct {
	found, missing []RemovedRun
}

func squash(s string) string { return strings.Join(strings.Fields(s), " ") }

func normalizeDocs(s string) string {
	var b strings.Builder
	for ln := range strings.SplitSeq(s, "\n") {
		ln = strings.TrimSpace(ln)
		ln = strings.TrimSpace(strings.TrimPrefix(ln, "//"))
		ln = strings.TrimSpace(strings.TrimPrefix(ln, ">"))
		b.WriteString(ln)
		b.WriteByte(' ')
	}
	return squash(b.String())
}

func movedTextReport(removed []RemovedRun, addedDocs string) movedReport {
	docs := normalizeDocs(addedDocs)
	var r movedReport
	for _, run := range removed {
		if strings.Contains(docs, squash(run.Text)) {
			r.found = append(r.found, run)
		} else {
			r.missing = append(r.missing, run)
		}
	}
	return r
}

func (r movedReport) write(w io.Writer, verbose bool) {
	total := func(rs []RemovedRun) int {
		n := 0
		for _, x := range rs {
			n += x.Lines
		}
		return n
	}
	fmt.Fprintf(w, "moved-text report: %d removed run(s), %d line(s): %d line(s) found verbatim in the docs this diff adds, %d line(s) NOT found (listed below, not red)\n",
		len(r.found)+len(r.missing), total(r.found)+total(r.missing), total(r.found), total(r.missing))
	show := func(label string, rs []RemovedRun) {
		for _, x := range rs {
			first, _, _ := strings.Cut(x.Text, "\n")
			if len(first) > 90 {
				first = first[:90] + "…"
			}
			fmt.Fprintf(w, "  %-9s %s:%d (%d line(s)) %q\n", label, x.File, x.Line, x.Lines, first)
		}
	}
	if verbose {
		show("found", r.found)
	}
	show("NOT FOUND", r.missing)
}

package main

import (
	"fmt"
	"io"
	"strings"
)

// The overall step of `gate comments-only` (comments_only.go), a grader of what its runner records: the changed paths, each
// modified .go file's compareComments verdict, the docs text the diff adds, the flags, and the results of the checks that
// follow a green.

// pathChange is one changed path and its git status letter.
type pathChange struct {
	Path   string `json:"path"`
	Status string `json:"status"`
}

// removedRun is a run of comment lines a file lost.
type removedRun struct {
	File  string `json:"file"`
	Line  int    `json:"line"`  // first line of the run in the old file
	Lines int    `json:"lines"` // comment lines in the run
	Text  string `json:"text"`  // the run, markers stripped, one line per source line
}

// fileVerdict is compareComments' answer for one modified .go file.
type fileVerdict struct {
	Reds            []string     `json:"reds,omitempty"`
	Removed         []removedRun `json:"removed,omitempty"`
	CommentLinesOld int          `json:"comment_lines_old"`
	CommentLinesNew int          `json:"comment_lines_new"`
}

// commentsOnlyFile is a modified .go file's verdict, or the red saying its two versions could not be read.
type commentsOnlyFile struct {
	Verdict    fileVerdict `json:"verdict"`
	Unreadable string      `json:"unreadable,omitempty"`
}

// commentsOnlyCheck is one check run after a green: gofmt (Kind "gofmt", red when it prints a file too) or a step.
type commentsOnlyCheck struct {
	Kind string `json:"kind"`
	Name string `json:"name"`
	OK   bool   `json:"ok"`
	Err  string `json:"err"` // the error as %v prints it ("<nil>" when it ran clean)
	Out  string `json:"out,omitempty"`
}

// commentsOnlyInput is what a comments-only run records for its grader.
type commentsOnlyInput struct {
	Changes  []pathChange                `json:"changes"`
	Files    map[string]commentsOnlyFile `json:"files,omitempty"`
	Added    string                      `json:"added_docs"`
	NoChecks bool                        `json:"no_checks,omitempty"`
	Verbose  bool                        `json:"verbose,omitempty"`
	Checks   []commentsOnlyCheck         `json:"checks,omitempty"`
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

// commentsOnlyReds is the comments check's findings over the recorded paths, in path order, and the modified .go files.
func commentsOnlyReds(in commentsOnlyInput) (reds, goFiles []string) {
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

// gradeCommentsOnly prints the report from the comment-line counts through the verdict, then the checks' lines when the
// comments check is green and checks were run, and returns the exit code.
func gradeCommentsOnly(w io.Writer, in commentsOnlyInput) int {
	reds, goFiles := commentsOnlyReds(in)
	var removed []removedRun
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
	return gradeCommentsOnlyChecks(w, in.Checks, len(goFiles))
}

// gradeCommentsOnlyChecks prints each check's line and the closing verdict.
func gradeCommentsOnlyChecks(w io.Writer, checks []commentsOnlyCheck, nGo int) int {
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
	for _, ln := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
		b.WriteString("      " + ln + "\n")
	}
	return b.String()
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
		if strings.Contains(docs, squash(run.Text)) {
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
			n += x.Lines
		}
		return n
	}
	fmt.Fprintf(w, "moved-text report: %d removed run(s), %d line(s): %d line(s) found verbatim in the docs this diff adds, %d line(s) NOT found (listed below, not red)\n",
		len(r.found)+len(r.missing), total(r.found)+total(r.missing), total(r.found), total(r.missing))
	show := func(label string, rs []removedRun) {
		for _, x := range rs {
			first := strings.SplitN(x.Text, "\n", 2)[0]
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

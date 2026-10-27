package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strings"

	"github.com/townsendmerino/goinfer/internal/paritykey"
)

// gate shadow: VK5's shadow mode (docs/tasks/task-subsystems-2026-10.md §5.8). For one push, from before to after, it
// classes every parity family by today's key and by the shadow key (the program builders out, the family's program
// fingerprints in; paritykey.ShadowKey) and gives each disagreement its reason. Both keys read only committed files, so
// any runner gives the same answer.
//
// Exit: 0 no unexplained disagreement (or VOID, which it prints), 1 an unexplained one, 2 misuse.

type shadowRecord struct {
	Before, After string
	Outcome       string
	Void          string `json:",omitempty"`
	Counts        map[paritykey.Class]int
	Verdicts      []paritykey.Verdict
}

func runShadow(argv []string, w io.Writer) int {
	fs := flag.NewFlagSet("gate shadow", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	jsonOut := fs.String("json", "", "also write the record as JSON to this file")
	if err := fs.Parse(argv); err != nil || fs.NArg() != 2 {
		fmt.Fprintln(os.Stderr, "usage: gate shadow [-json FILE] <before> <after>")
		return 2
	}
	root, err := repoRoot()
	if err != nil {
		fmt.Fprintln(os.Stderr, "gate shadow:", err)
		return 2
	}
	rec := shadowRecord{Before: fs.Arg(0), After: fs.Arg(1)}
	code := shadow(root, &rec)
	writeShadow(w, rec)
	if *jsonOut != "" {
		b, _ := json.MarshalIndent(rec, "", "  ")
		if err := os.WriteFile(*jsonOut, b, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "gate shadow:", err)
			return 2
		}
	}
	return code
}

var zeroSHA = regexp.MustCompile(`^0+$`)

func shadow(root string, rec *shadowRecord) int {
	void := func(why string) int {
		rec.Outcome, rec.Void = "VOID", why
		return 0
	}
	for _, rev := range []*string{&rec.Before, &rec.After} {
		if zeroSHA.MatchString(*rev) {
			return void("no before commit (a new branch)")
		}
		out, err := exec.Command("git", "-C", root, "rev-parse", "--verify", "--quiet", *rev+"^{commit}").Output()
		if err != nil {
			return void(*rev + " is not a commit here (a force push, or a shallow clone)")
		}
		*rev = strings.TrimSpace(string(out))
	}
	g, err := paritykey.OpenGit(root)
	if err != nil {
		return void(err.Error())
	}
	defer g.Close()
	before, err := paritykey.LoadSide(g.At(rec.Before))
	if err != nil {
		return void("before: " + err.Error())
	}
	after, err := paritykey.LoadSide(g.At(rec.After))
	if err != nil {
		return void("after: " + err.Error())
	}
	rec.Verdicts = paritykey.Compare(before, after)
	rec.Counts = map[paritykey.Class]int{}
	unexplained := 0
	for _, v := range rec.Verdicts {
		rec.Counts[v.Class]++
		if !v.Explained {
			unexplained++
		}
	}
	if unexplained > 0 {
		rec.Outcome = fmt.Sprintf("UNEXPLAINED (%d)", unexplained)
		return 1
	}
	rec.Outcome = "OK"
	return 0
}

// writeShadow prints the record as markdown, for a CI job summary.
func writeShadow(w io.Writer, rec shadowRecord) {
	short := func(s string) string { return s[:min(len(s), 8)] }
	fmt.Fprintf(w, "## VK5 shadow: %s..%s — %s\n\n", short(rec.Before), short(rec.After), rec.Outcome)
	if rec.Void != "" {
		fmt.Fprintf(w, "VOID: %s\n", rec.Void)
		return
	}
	fmt.Fprintf(w, "| class | families |\n|---|---:|\n")
	for _, c := range []paritykey.Class{paritykey.AgreeStale, paritykey.AgreeFresh, paritykey.OldOnly, paritykey.NewOnly, paritykey.Added, paritykey.Removed} {
		fmt.Fprintf(w, "| %s | %d |\n", c, rec.Counts[c])
	}
	var dis []paritykey.Verdict
	for _, v := range rec.Verdicts {
		if v.Class == paritykey.OldOnly || v.Class == paritykey.NewOnly {
			dis = append(dis, v)
		}
	}
	if len(dis) == 0 {
		return
	}
	fmt.Fprintf(w, "\n| family | class | explained | moved | reason |\n|---|---|---|---|---|\n")
	for _, v := range dis {
		fmt.Fprintf(w, "| %s | %s | %v | %s | %s |\n", v.Family, v.Class, v.Explained, strings.Join(v.Moved, ", "), v.Reason)
	}
}

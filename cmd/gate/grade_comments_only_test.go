package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

// recordedCommentsOnly is the comments-only-replay golden's recorded input.
func recordedCommentsOnly(t *testing.T) commentsOnlyInput {
	t.Helper()
	rec, err := readRunRecord(filepath.Join(goldenRoot, "comments-only-replay", "gate_comments-only_run.json"))
	if err != nil {
		t.Fatal(err)
	}
	return *rec.CommentsOnly
}

// TestGradeCommentsOnly_plants is SB12-C4 for step 13: from the recorded run, a path outside docs/ that is not Go, an added
// .go file and a file verdict with a red each make the run red and print that no check ran; a failed gofmt, vet or lint on a
// green comments check each say so and exit 1; -no-checks stops at the comments verdict.
func TestGradeCommentsOnly_plants(t *testing.T) {
	for name, tc := range map[string]struct {
		edit func(in *commentsOnlyInput)
		code int
		want []string
	}{
		"as recorded": {func(*commentsOnlyInput) {}, 0, []string{"GREEN: comments-only, gofmt, go vet and the citation lint"}},
		"a path that is neither Go nor docs": {func(in *commentsOnlyInput) {
			in.Changes = append(in.Changes, pathChange{Path: "scripts/x.py", Status: "M"})
		}, 1, []string{"scripts/x.py: neither a .go file nor under docs/", "not run: gofmt, go vet and the citation lint"}},
		"an added .go file": {func(in *commentsOnlyInput) {
			in.Changes = append(in.Changes, pathChange{Path: "cmd/gate/new.go", Status: "A"})
		}, 1, []string{"cmd/gate/new.go: a .go file was added"}},
		"a file verdict with a red": {func(in *commentsOnlyInput) {
			f := in.Files["cmd/gate/gpu_test.go"]
			f.Verdict.Reds = []string{"cmd/gate/gpu_test.go: CODE CHANGED — a token"}
			in.Files["cmd/gate/gpu_test.go"] = f
		}, 1, []string{"RED: 1 finding(s)", "CODE CHANGED — a token"}},
		"a failed gofmt": {func(in *commentsOnlyInput) {
			in.Checks[0].Out = "cmd/gate/gpu_test.go\n"
		}, 1, []string{"RED   gofmt -l: <nil>\n      cmd/gate/gpu_test.go", "a following check failed"}},
		"a failed vet": {func(in *commentsOnlyInput) {
			in.Checks[1].OK, in.Checks[1].Err, in.Checks[1].Out = false, "exit status 1", "vet: x.go:1: bad\n"
		}, 1, []string{"RED   go vet cmd/gate: exit status 1\n      vet: x.go:1: bad", "a following check failed"}},
		"a red lint": {func(in *commentsOnlyInput) {
			last := len(in.Checks) - 1
			in.Checks[last].OK, in.Checks[last].Err = false, "exit status 1"
		}, 1, []string{"RED   citation lint: exit status 1"}},
		"-no-checks": {func(in *commentsOnlyInput) { in.NoChecks = true }, 0, []string{"GREEN: 1 .go file(s) differ only in comments"}},
	} {
		in := recordedCommentsOnly(t)
		tc.edit(&in)
		var out bytes.Buffer
		code := gradeCommentsOnly(&out, in)
		ok := code == tc.code
		for _, s := range tc.want {
			ok = ok && strings.Contains(out.String(), s)
		}
		if name == "-no-checks" && strings.Contains(out.String(), "gofmt") {
			ok = false
		}
		if !ok {
			t.Errorf("%s: exit %d, want %d naming %q:\n%s", name, code, tc.code, tc.want, out.String())
		}
	}
}

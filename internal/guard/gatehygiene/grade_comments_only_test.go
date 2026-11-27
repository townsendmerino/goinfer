package gatehygiene

import (
	"bytes"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
	"github.com/townsendmerino/goinfer/internal/testpath"
)

func recordedCommentsOnly(t *testing.T) CommentsOnlyInput {
	t.Helper()
	rec, err := gatecore.ReadRunRecord(testpath.Testdata(t, "gate", "goldens", "comments-only-replay", "gate_comments-only_run.json"))
	if err != nil {
		t.Fatal(err)
	}
	co, err := gatecore.UnmarshalPayload[CommentsOnlyInput](rec.CommentsOnly)
	if err != nil || co == nil {
		t.Fatalf("comments_only payload: %v", err)
	}
	return *co
}

func TestGradeCommentsOnly_plants(t *testing.T) {
	for name, tc := range map[string]struct {
		edit func(in *CommentsOnlyInput)
		code int
		want []string
	}{
		"as recorded": {func(*CommentsOnlyInput) {}, 0, []string{"GREEN: comments-only, gofmt, go vet and the citation lint"}},
		"a path that is neither Go nor docs": {func(in *CommentsOnlyInput) {
			in.Changes = append(in.Changes, PathChange{Path: "scripts/x.py", Status: "M"})
		}, 1, []string{"scripts/x.py: neither a .go file nor under docs/", "not run: gofmt, go vet and the citation lint"}},
		"an added .go file": {func(in *CommentsOnlyInput) {
			in.Changes = append(in.Changes, PathChange{Path: "cmd/gate/new.go", Status: "A"})
		}, 1, []string{"cmd/gate/new.go: a .go file was added"}},
		"a file verdict with a red": {func(in *CommentsOnlyInput) {
			f := in.Files["cmd/gate/gpu_test.go"]
			f.Verdict.Reds = []string{"cmd/gate/gpu_test.go: CODE CHANGED — a token"}
			in.Files["cmd/gate/gpu_test.go"] = f
		}, 1, []string{"RED: 1 finding(s)", "CODE CHANGED — a token"}},
		"a failed gofmt": {func(in *CommentsOnlyInput) {
			in.Checks[0].Out = "cmd/gate/gpu_test.go\n"
		}, 1, []string{"RED   gofmt -l: <nil>\n      cmd/gate/gpu_test.go", "a following check failed"}},
		"a failed vet": {func(in *CommentsOnlyInput) {
			in.Checks[1].OK, in.Checks[1].Err, in.Checks[1].Out = false, "exit status 1", "vet: x.go:1: bad\n"
		}, 1, []string{"RED   go vet cmd/gate: exit status 1\n      vet: x.go:1: bad", "a following check failed"}},
		"a red lint": {func(in *CommentsOnlyInput) {
			last := len(in.Checks) - 1
			in.Checks[last].OK, in.Checks[last].Err = false, "exit status 1"
		}, 1, []string{"RED   citation lint: exit status 1"}},
		"-no-checks": {func(in *CommentsOnlyInput) { in.NoChecks = true }, 0, []string{"GREEN: 1 .go file(s) differ only in comments"}},
	} {
		in := recordedCommentsOnly(t)
		tc.edit(&in)
		var out bytes.Buffer
		code := GradeCommentsOnly(&out, in)
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

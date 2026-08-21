package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The mutation checker is a gate's own gate, so it needs its own — otherwise the mechanism built to
// prove gates are falsifiable is itself unfalsified. Each case below is a way this tool could
// certify a gate as falsifiable while nothing was exercised, which is the exact defect class its
// header records.

// scratchRepo makes a git repo containing one subject file, and chdirs into it so repoRoot() lands
// there. It returns the subject path relative to the repo, as the CLI takes it.
func scratchRepo(t *testing.T, body string) (dir, rel string) {
	t.Helper()
	for _, bin := range []string{"git", "sed", "grep"} {
		if _, err := exec.LookPath(bin); err != nil {
			t.Skipf("%s not on PATH", bin)
		}
	}
	dir = t.TempDir()
	// macOS hands out /var symlinks for TempDir; git reports the resolved path, and the chdir
	// comparison below would disagree with itself without this.
	if r, err := filepath.EvalSymlinks(dir); err == nil {
		dir = r
	}
	cmd := exec.Command("git", "init", "-q")
	cmd.Dir = dir
	if err := cmd.Run(); err != nil {
		t.Skipf("git init: %v", err)
	}
	rel = "subject.txt"
	if err := os.WriteFile(filepath.Join(dir, rel), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	// runMutation chdirs to the repo root itself; put the caller back either way.
	t.Cleanup(func() { _ = os.Chdir(wd) })
	return dir, rel
}

func subjectBytes(t *testing.T, dir, rel string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, rel))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

const subjectOK = "STATE=OK\nsecond line\n"

// The happy path: green -> mutate -> red -> restore -> green.
func TestMutation_falsifiableGatePasses(t *testing.T) {
	dir, rel := scratchRepo(t, subjectOK)
	var buf strings.Builder
	rc := runMutation([]string{"demo", rel, "s/STATE=OK/STATE=BROKEN/",
		"grep", "-q", "STATE=OK", rel}, &buf)
	out := buf.String()
	if rc != 0 {
		t.Fatalf("rc=%d, want 0\n%s", rc, out)
	}
	for _, want := range []string{"baseline green", "mutation applied", "red under mutation", "green after restore", "is falsifiable"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
	if got := subjectBytes(t, dir, rel); got != subjectOK {
		t.Errorf("the subject file was not restored:\n%q", got)
	}
}

// A sed expression that matches nothing leaves a green run that LOOKS like a verified mutation
// check. It happened for real — float32(v/sc) where both operands were already float32.
func TestMutation_vacuousExpressionIsRejected(t *testing.T) {
	dir, rel := scratchRepo(t, subjectOK)
	var buf strings.Builder
	rc := runMutation([]string{"vacuous", rel, "s/NOT-PRESENT/ALSO-NOT/",
		"grep", "-q", "STATE=OK", rel}, &buf)
	if rc != 1 {
		t.Fatalf("rc=%d, want 1 — a no-op mutation must not certify anything\n%s", rc, buf.String())
	}
	if !strings.Contains(buf.String(), "changed NOTHING") {
		t.Errorf("the report does not say why:\n%s", buf.String())
	}
	if got := subjectBytes(t, dir, rel); got != subjectOK {
		t.Errorf("file not restored:\n%q", got)
	}
}

// A gate that is ALREADY RED proves nothing by going red under mutation — the mutation would not be
// what turned it. Checked FIRST, before anything is written to the subject file.
func TestMutation_alreadyRedBaselineIsRejected(t *testing.T) {
	dir, rel := scratchRepo(t, subjectOK)
	var buf strings.Builder
	rc := runMutation([]string{"already-red", rel, "s/STATE=OK/STATE=BROKEN/",
		"grep", "-q", "MARKER-THAT-IS-ABSENT", rel}, &buf)
	out := buf.String()
	if rc != 1 {
		t.Fatalf("rc=%d, want 1\n%s", rc, out)
	}
	if !strings.Contains(out, "ALREADY RED") {
		t.Errorf("the report does not name the reason:\n%s", out)
	}
	if strings.Contains(out, "mutation applied") {
		t.Errorf("it mutated the file despite a red baseline:\n%s", out)
	}
	if got := subjectBytes(t, dir, rel); got != subjectOK {
		t.Errorf("file touched on the baseline path:\n%q", got)
	}
}

// The finding this whole tool exists to produce: the gate does not see what it claims to check.
func TestMutation_gateThatSurvivesMutationIsReported(t *testing.T) {
	dir, rel := scratchRepo(t, subjectOK)
	var buf strings.Builder
	// grep for "STATE=" — true both before and after, so the gate is blind to this mutation.
	rc := runMutation([]string{"blind", rel, "s/STATE=OK/STATE=BROKEN/",
		"grep", "-q", "STATE=", rel}, &buf)
	if rc != 1 {
		t.Fatalf("rc=%d, want 1\n%s", rc, buf.String())
	}
	if !strings.Contains(buf.String(), "cannot see the thing it claims to check") {
		t.Errorf("the finding is not stated:\n%s", buf.String())
	}
	if got := subjectBytes(t, dir, rel); got != subjectOK {
		t.Errorf("file not restored after a failed check:\n%q", got)
	}
}

// Misuse is exit 2, distinct from a finding (1) and from a pass (0) — a caller must be able to tell
// "the check ran and disagreed" from "the check could not run".
func TestMutation_misuseIsExitTwo(t *testing.T) {
	_, rel := scratchRepo(t, subjectOK)
	var buf strings.Builder
	if rc := runMutation([]string{"short"}, &buf); rc != 2 {
		t.Errorf("too few args: rc=%d, want 2", rc)
	}
	if rc := runMutation([]string{"nofile", "no/such/file.go", "s/a/b/", "grep", "-q", "x", rel}, &buf); rc != 2 {
		t.Errorf("missing subject file: rc=%d, want 2", rc)
	}
}

// A verify command that cannot START must not read as green. `go test` on a nonexistent binary,
// a typo'd tool name — the shell's `"$@"` would have produced 127, and a bare `if cmd.Run() == nil`
// in Go would have produced a non-nil error handled as "red", which is right, but the baseline step
// must treat it as red rather than as a green baseline.
func TestMutation_unstartableVerifyCommandIsNotGreen(t *testing.T) {
	_, rel := scratchRepo(t, subjectOK)
	var buf strings.Builder
	rc := runMutation([]string{"nocmd", rel, "s/STATE=OK/STATE=BROKEN/",
		"definitely-not-a-real-binary-xyz"}, &buf)
	if rc != 1 {
		t.Fatalf("rc=%d, want 1 — an unstartable verify command is not a green baseline\n%s", rc, buf.String())
	}
	if !strings.Contains(buf.String(), "ALREADY RED") {
		t.Errorf("expected the baseline step to reject it:\n%s", buf.String())
	}
}

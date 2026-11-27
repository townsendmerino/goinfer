package gatehygiene

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func scratchRepo(t *testing.T, body string) (dir, rel string) {
	t.Helper()
	for _, bin := range []string{"git", "sed", "grep"} {
		if _, err := exec.LookPath(bin); err != nil {
			t.Skipf("%s not on PATH", bin)
		}
	}
	dir = t.TempDir()
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
	t.Chdir(dir)
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

func TestMutation_falsifiableGatePasses(t *testing.T) {
	dir, rel := scratchRepo(t, subjectOK)
	var buf strings.Builder
	rc := RunMutation([]string{"demo", rel, "s/STATE=OK/STATE=BROKEN/",
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

func TestMutation_vacuousExpressionIsRejected(t *testing.T) {
	dir, rel := scratchRepo(t, subjectOK)
	var buf strings.Builder
	rc := RunMutation([]string{"vacuous", rel, "s/NOT-PRESENT/ALSO-NOT/",
		"grep", "-q", "STATE=OK", rel}, &buf)
	if rc != 1 {
		t.Fatalf("rc=%d, want 1\n%s", rc, buf.String())
	}
	if !strings.Contains(buf.String(), "changed NOTHING") {
		t.Errorf("the report does not say why:\n%s", buf.String())
	}
	if got := subjectBytes(t, dir, rel); got != subjectOK {
		t.Errorf("file not restored:\n%q", got)
	}
}

func TestMutation_alreadyRedBaselineIsRejected(t *testing.T) {
	dir, rel := scratchRepo(t, subjectOK)
	var buf strings.Builder
	rc := RunMutation([]string{"already-red", rel, "s/STATE=OK/STATE=BROKEN/",
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

func TestMutation_gateThatSurvivesMutationIsReported(t *testing.T) {
	dir, rel := scratchRepo(t, subjectOK)
	var buf strings.Builder
	rc := RunMutation([]string{"blind", rel, "s/STATE=OK/STATE=BROKEN/",
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

func TestMutation_misuseIsExitTwo(t *testing.T) {
	_, rel := scratchRepo(t, subjectOK)
	var buf strings.Builder
	if rc := RunMutation([]string{"short"}, &buf); rc != 2 {
		t.Errorf("too few args: rc=%d, want 2", rc)
	}
	if rc := RunMutation([]string{"nofile", "no/such/file.go", "s/a/b/", "grep", "-q", "x", rel}, &buf); rc != 2 {
		t.Errorf("missing subject file: rc=%d, want 2", rc)
	}
}

func TestMutation_unstartableVerifyCommandIsNotGreen(t *testing.T) {
	_, rel := scratchRepo(t, subjectOK)
	var buf strings.Builder
	rc := RunMutation([]string{"nocmd", rel, "s/STATE=OK/STATE=BROKEN/",
		"definitely-not-a-real-binary-xyz"}, &buf)
	if rc != 1 {
		t.Fatalf("rc=%d, want 1\n%s", rc, buf.String())
	}
	if !strings.Contains(buf.String(), "ALREADY RED") {
		t.Errorf("expected the baseline step to reject it:\n%s", buf.String())
	}
}

func TestMutation_sedInvocationIsPortable(t *testing.T) {
	args := sedArgs("s/a/b/", "some/file.go")
	for _, a := range args {
		if a == "-i" || strings.HasPrefix(a, "-i") {
			t.Fatalf("sed invocation uses %q: %v", a, args)
		}
	}
	if len(args) != 2 || args[0] != "s/a/b/" || args[1] != "some/file.go" {
		t.Fatalf("sedArgs = %v, want [expr file]", args)
	}
}

func TestMutation_failedRestoreIsReportedAndKeepsTheBackup(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root writes through a read-only mode, so the restore cannot be made to fail")
	}
	dir, rel := scratchRepo(t, subjectOK)
	t.Cleanup(func() { _ = os.Chmod(filepath.Join(dir, rel), 0o644) })
	var buf strings.Builder
	rc := RunMutation([]string{"lockout", rel, "s/STATE=OK/STATE=BROKEN/",
		"sh", "-c", "if grep -q BROKEN " + rel + "; then chmod 0444 " + rel + "; fi; exit 0"}, &buf)
	out := buf.String()
	if rc != 2 {
		t.Errorf("rc=%d, want 2\n%s", rc, out)
	}
	if !strings.Contains(out, "NOT restored") {
		t.Errorf("the failed restore is not reported:\n%s", out)
	}
	bak := filepath.Join(os.TempDir(), "gate_mutation_lockout.bak")
	t.Cleanup(func() { _ = os.Remove(bak) })
	if b, err := os.ReadFile(bak); err != nil || string(b) != subjectOK {
		t.Errorf("the backup is gone or wrong (%v): %q", err, b)
	}
}

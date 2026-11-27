package gaterelease

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
)

// The GPU gate's records and the rule that decides whether one still covers a later commit (gpu_record.go). The rule is tested against a
// real git repository built in the test, because it is a statement about git trees.

func gitIn(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-c", "user.name=t", "-c", "user.email=t@t"}, args...)...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

// gpuTestRepo is a repository with the four qualifying directories and a docs/ directory, and a commit function.
func gpuTestRepo(t *testing.T) (dir string, commit func(file, content string) string) {
	t.Helper()
	dir = t.TempDir()
	gitIn(t, dir, "init", "-q", "-b", "main")
	write := func(file, content string) {
		p := filepath.Join(dir, file)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, d := range append([]string{"docs"}, GPUQualifyingTrees...) {
		write(d+"/f.txt", "v0")
	}
	gitIn(t, dir, "add", ".")
	gitIn(t, dir, "commit", "-q", "-m", "base")
	n := 0
	return dir, func(file, content string) string {
		n++
		write(file, content+strings.Repeat("x", n))
		gitIn(t, dir, "add", ".")
		gitIn(t, dir, "commit", "-q", "-m", "change "+file)
		return gitIn(t, dir, "rev-parse", "HEAD")
	}
}

func treeOfIn(t *testing.T, dir string) func(rev, d string) (string, error) {
	return func(rev, d string) (string, error) { return gatecore.GitTree(dir, rev, d) }
}

func rec(backend, commit, date, verdict string) GPURecord {
	return GPURecord{Schema: 1, Backend: backend, Commit: commit, Date: date, Verdict: verdict}
}

// A run covers a later commit that changed docs, and does not cover one that changed any of the four directories.
func TestGPURecord_coversLaterCommitsOnlyIfTheCodeIsUnchanged(t *testing.T) {
	dir, commit := gpuTestRepo(t)
	ran := gitIn(t, dir, "rev-parse", "HEAD")
	docs := commit("docs/f.txt", "doc edit")
	treeOf := treeOfIn(t, dir)
	recs := []GPURecord{rec("cuda", ran, "2026-10-11T10:00:00Z", "PASS")}

	if r, why := GPUCoverage(recs, "cuda", docs, treeOf); r == nil {
		t.Fatalf("a docs-only change voided the run: %s", why)
	}
	for _, d := range []string{"backend/metal", "backend/cuda", "backend/webgpu", "decoder"} { // RELEASING.md's four, spelled out so the rule cannot be edited out from under its test
		gitIn(t, dir, "checkout", "-q", "-B", "from-"+d, ran) // each change on its own, from the commit the run was made at
		head := commit(d+"/f.txt", "code edit")
		r, why := GPUCoverage(recs, "cuda", head, treeOf)
		if r != nil || !strings.Contains(why, d+"/ changed since") {
			t.Errorf("a change under %s/ left the run covering the commit (%v, %q)", d, r, why)
		}
	}
	if r, why := GPUCoverage(recs, "metal", docs, treeOf); r != nil || why != "no record" {
		t.Errorf("metal is covered by a cuda record: %v %q", r, why)
	}
}

// Only a PASS on a clean tree with the heavy tier run counts; a run that is none of those at the same code is named, not skipped.
func TestGPURecord_onlyACompletePassCounts(t *testing.T) {
	dir, _ := gpuTestRepo(t)
	head := gitIn(t, dir, "rev-parse", "HEAD")
	treeOf := treeOfIn(t, dir)
	for name, mod := range map[string]func(*GPURecord){
		"FAIL":         func(r *GPURecord) { r.Verdict = "FAIL" },
		"INCONCLUSIVE": func(r *GPURecord) { r.Verdict = "INCONCLUSIVE" },
		"dirty":        func(r *GPURecord) { r.Dirty = true },
		"heavy":        func(r *GPURecord) { r.HeavySkipped = true },
		"narrowed":     func(r *GPURecord) { r.HeavyGroups = []string{"kernels", "qwen3-4b"} },
	} {
		r := rec("cuda", head, "2026-10-11T10:00:00Z", "PASS")
		mod(&r)
		got, why := GPUCoverage([]GPURecord{r}, "cuda", head, treeOf)
		if got != nil || !strings.Contains(why, "does not count") {
			t.Errorf("%s: covered=%v why=%q", name, got != nil, why)
		}
		if name == "narrowed" && !strings.Contains(why, "narrowed to kernels, qwen3-4b") {
			t.Errorf("a narrowed record's refusal does not name the narrowing: %q", why)
		}
	}
	if got, _ := GPUCoverage([]GPURecord{rec("cuda", head, "2026-10-11T10:00:00Z", "PASS")}, "cuda", head, treeOf); got == nil {
		t.Error("a clean complete PASS at HEAD does not cover HEAD")
	}
}

// A later FAIL at unchanged code is not hidden by an earlier PASS, and a later PASS at changed code does not hide nothing: the newest
// record at the SAME code decides.
func TestGPURecord_theNewestRecordAtTheSameCodeDecides(t *testing.T) {
	dir, commit := gpuTestRepo(t)
	a := gitIn(t, dir, "rev-parse", "HEAD")
	b := commit("docs/f.txt", "docs")
	treeOf := treeOfIn(t, dir)
	recs := []GPURecord{rec("cuda", a, "2026-10-11T10:00:00Z", "PASS"), rec("cuda", b, "2026-10-11T11:00:00Z", "FAIL")}
	if r, why := GPUCoverage(recs, "cuda", b, treeOf); r != nil || !strings.Contains(why, "verdict FAIL") {
		t.Errorf("an earlier PASS hid a later FAIL at the same code: %v %q", r, why)
	}
	recs[1].Verdict = "PASS"
	if r, _ := GPUCoverage(recs, "cuda", b, treeOf); r == nil {
		t.Error("two passes at the same code do not cover it")
	}
}

// An unreadable record is an error, and a stray file is not a record.
func TestGPURecord_loadRefusesAMalformedRecord(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README.txt"), []byte("not a record"), 0o644); err != nil {
		t.Fatal(err)
	}
	if recs, err := LoadGPURecords(dir); err != nil || len(recs) != 0 {
		t.Fatalf("a non-JSON file: %v %v", recs, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "cuda-x.json"), []byte(`{"schema":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadGPURecords(dir); err == nil {
		t.Fatal("a record with no backend or commit was accepted")
	}
	if recs, err := LoadGPURecords(filepath.Join(dir, "missing")); err != nil || recs != nil {
		t.Fatalf("a missing directory is no records, not an error: %v %v", recs, err)
	}
}

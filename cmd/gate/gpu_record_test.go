package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
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
	for _, d := range append([]string{"docs"}, gpuQualifyingTrees...) {
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
	return func(rev, d string) (string, error) { return gitTree(dir, rev, d) }
}

func rec(backend, commit, date, verdict string) gpuRecord {
	return gpuRecord{Schema: 1, Backend: backend, Commit: commit, Date: date, Verdict: verdict}
}

// A run covers a later commit that changed docs, and does not cover one that changed any of the four directories.
func TestGPURecord_coversLaterCommitsOnlyIfTheCodeIsUnchanged(t *testing.T) {
	dir, commit := gpuTestRepo(t)
	ran := gitIn(t, dir, "rev-parse", "HEAD")
	docs := commit("docs/f.txt", "doc edit")
	treeOf := treeOfIn(t, dir)
	recs := []gpuRecord{rec("cuda", ran, "2026-10-11T10:00:00Z", "PASS")}

	if r, why := gpuCoverage(recs, "cuda", docs, treeOf); r == nil {
		t.Fatalf("a docs-only change voided the run: %s", why)
	}
	for _, d := range []string{"metal", "cuda", "gpu", "decoder"} { // RELEASING.md's four, spelled out so the rule cannot be edited out from under its test
		gitIn(t, dir, "checkout", "-q", "-B", "from-"+d, ran) // each change on its own, from the commit the run was made at
		head := commit(d+"/f.txt", "code edit")
		r, why := gpuCoverage(recs, "cuda", head, treeOf)
		if r != nil || !strings.Contains(why, d+"/ changed since") {
			t.Errorf("a change under %s/ left the run covering the commit (%v, %q)", d, r, why)
		}
	}
	if r, why := gpuCoverage(recs, "metal", docs, treeOf); r != nil || why != "no record" {
		t.Errorf("metal is covered by a cuda record: %v %q", r, why)
	}
}

// Only a PASS on a clean tree with the heavy tier run counts; a run that is none of those at the same code is named, not skipped.
func TestGPURecord_onlyACompletePassCounts(t *testing.T) {
	dir, _ := gpuTestRepo(t)
	head := gitIn(t, dir, "rev-parse", "HEAD")
	treeOf := treeOfIn(t, dir)
	for name, mod := range map[string]func(*gpuRecord){
		"FAIL":         func(r *gpuRecord) { r.Verdict = "FAIL" },
		"INCONCLUSIVE": func(r *gpuRecord) { r.Verdict = "INCONCLUSIVE" },
		"dirty":        func(r *gpuRecord) { r.Dirty = true },
		"heavy":        func(r *gpuRecord) { r.HeavySkipped = true },
	} {
		r := rec("cuda", head, "2026-10-11T10:00:00Z", "PASS")
		mod(&r)
		got, why := gpuCoverage([]gpuRecord{r}, "cuda", head, treeOf)
		if got != nil || !strings.Contains(why, "does not count") {
			t.Errorf("%s: covered=%v why=%q", name, got != nil, why)
		}
	}
	if got, _ := gpuCoverage([]gpuRecord{rec("cuda", head, "2026-10-11T10:00:00Z", "PASS")}, "cuda", head, treeOf); got == nil {
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
	recs := []gpuRecord{rec("cuda", a, "2026-10-11T10:00:00Z", "PASS"), rec("cuda", b, "2026-10-11T11:00:00Z", "FAIL")}
	if r, why := gpuCoverage(recs, "cuda", b, treeOf); r != nil || !strings.Contains(why, "verdict FAIL") {
		t.Errorf("an earlier PASS hid a later FAIL at the same code: %v %q", r, why)
	}
	recs[1].Verdict = "PASS"
	if r, _ := gpuCoverage(recs, "cuda", b, treeOf); r == nil {
		t.Error("two passes at the same code do not cover it")
	}
}

// An unreadable record is an error, and a stray file is not a record.
func TestGPURecord_loadRefusesAMalformedRecord(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "README.txt"), []byte("not a record"), 0o644); err != nil {
		t.Fatal(err)
	}
	if recs, err := loadGPURecords(dir); err != nil || len(recs) != 0 {
		t.Fatalf("a non-JSON file: %v %v", recs, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "cuda-x.json"), []byte(`{"schema":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadGPURecords(dir); err == nil {
		t.Fatal("a record with no backend or commit was accepted")
	}
	if recs, err := loadGPURecords(filepath.Join(dir, "missing")); err != nil || recs != nil {
		t.Fatalf("a missing directory is no records, not an error: %v %v", recs, err)
	}
}

// The verdict block writes the record, with the verdict word, the counts and the tree ids; the heavy skip and a dirty tree are in it.
func TestGPURecord_theVerdictWritesTheRecord(t *testing.T) {
	for _, tc := range []struct {
		name         string
		fail         int
		dirty, heavy bool
		want         string
	}{
		{"pass", 0, false, false, "PASS"},
		{"fail", 1, false, false, "FAIL"},
		{"dirty", 0, true, false, "INCONCLUSIVE"},
		{"heavy skipped still a PASS verdict, flagged", 0, false, true, "PASS"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			var out bytes.Buffer
			g := &gpuGate{w: &out, backend: "cuda", commit: "abc", dirty: tc.dirty, heavySkipped: tc.heavy, recordDir: dir, gpuName: "test gpu",
				expect: []string{"seam"}, emitted: map[string]bool{"seam": true}, pass: 3, fail: tc.fail, ran: 3}
			g.verdict()
			ents, _ := os.ReadDir(dir)
			if len(ents) != 1 {
				t.Fatalf("%d record files written, want 1 (output: %s)", len(ents), out.String())
			}
			b, _ := os.ReadFile(filepath.Join(dir, ents[0].Name()))
			var r gpuRecord
			if err := json.Unmarshal(b, &r); err != nil {
				t.Fatal(err)
			}
			if r.Verdict != tc.want || r.Dirty != tc.dirty || r.HeavySkipped != tc.heavy || r.Groups.Pass != 3 || r.Backend != "cuda" || len(r.Commit) != 40 {
				t.Errorf("record = %+v, want verdict %s dirty %v heavy %v", r, tc.want, tc.dirty, tc.heavy)
			}
			if len(r.Trees) != len(gpuQualifyingTrees) {
				t.Errorf("trees = %v, want an id for each of %v", r.Trees, gpuQualifyingTrees)
			}
		})
	}
}

// With no record directory the run writes nothing.
func TestGPURecord_noDirectoryWritesNothing(t *testing.T) {
	var out bytes.Buffer
	g := &gpuGate{w: &out, backend: "cuda", expect: []string{"seam"}, emitted: map[string]bool{"seam": true}, pass: 1, ran: 1}
	g.verdict()
	if strings.Contains(out.String(), "record:") || strings.Contains(out.String(), "RECORD NOT WRITTEN") {
		t.Errorf("a run with no record directory mentioned a record: %s", out.String())
	}
}

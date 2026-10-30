package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/paritykey"
)

// The parity sweep's record and the release row that reads it (parity_record.go), registered here before the code: no
// record reads OWED, a record made at code that has since changed reads OWED and names what changed, and a complete sweep
// at the same code reads done. "The same code" is every file the parity manifest hashes plus decoder/, cuda/, metal/ and
// gpu/. The rule is tested against a real git repository built in the test, because it is a statement about git objects.

// parityTestManifest is the real manifest's shape (families keyed by name; TestParityRecord_readsTheCommittedManifest holds
// the parser to the committed file itself). It hashes files inside a qualifying directory and files outside every one of them.
const parityTestManifest = `{"shared_sets": {"core": ["decoder/a.go", "scripts/pin.py"]},
 "families": {"fam": {"uses": ["core"], "own": ["decoder/b.go", "scripts/own.py"], "deps_hash": "sha256:x"}}}`

// parityRecordsTree is a manifest in the single-object shape, written as records (paritykey.RecordsDir).
func parityRecordsTree(t *testing.T, manifest string) map[string]string {
	t.Helper()
	var m struct {
		SharedSets json.RawMessage            `json:"shared_sets"`
		Families   map[string]json.RawMessage `json:"families"`
	}
	if err := json.Unmarshal([]byte(manifest), &m); err != nil {
		t.Fatal(err)
	}
	out := map[string]string{paritykey.SharedSetsFile: string(m.SharedSets)}
	for name, f := range m.Families {
		out[paritykey.RecordsDir+"/"+name+".json"] = string(f)
	}
	return out
}

// parityTestRepo is gpuTestRepo plus a parity manifest, the two scripts it hashes and one it does not.
func parityTestRepo(t *testing.T) (dir string, commit func(file, content string) string) {
	t.Helper()
	dir, commit = gpuTestRepo(t)
	writeTree(t, dir, parityRecordsTree(t, parityTestManifest))
	writeTree(t, dir, map[string]string{"scripts/pin.py": "pin v0", "scripts/own.py": "own v0", "scripts/unrelated.py": "u v0"})
	gitIn(t, dir, "add", ".")
	gitIn(t, dir, "commit", "-q", "-m", "manifest and scripts")
	return dir, commit
}

func parityPathsIn(t *testing.T, dir string) []string {
	t.Helper()
	b, err := paritykey.ManifestJSON(paritykey.Dir(dir))
	if err != nil {
		t.Fatal(err)
	}
	paths, err := parityQualifyingPaths(b)
	if err != nil {
		t.Fatal(err)
	}
	return paths
}

func objOfIn(dir string) func(rev, path string) (string, error) {
	return func(rev, path string) (string, error) { return gitObject(dir, rev, path) }
}

func prec(commit, date, verdict string) parityRecord {
	return parityRecord{Schema: 1, Commit: commit, Date: date, Verdict: verdict, Realckpt: true}
}

// The qualifying set is the four directories and every hashed file that is outside them, each once.
func TestParityRecord_qualifyingPaths(t *testing.T) {
	got, err := parityQualifyingPaths([]byte(parityTestManifest))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"cuda", "decoder", "gpu", "metal", "scripts/own.py", "scripts/pin.py"}
	if !slices.Equal(got, want) {
		t.Errorf("qualifying paths %v, want %v", got, want)
	}
	if _, err := parityQualifyingPaths([]byte(`{"families": {}}`)); err == nil {
		t.Error("a manifest that hashes no file was accepted: the row would then cover nothing but the four directories without saying so")
	}
	if _, err := parityQualifyingPaths([]byte(`not json`)); err == nil {
		t.Error("an unreadable manifest was accepted")
	}
}

// The parser reads the manifest this repository commits, not only the test's own: every path it returns exists, the four
// directories are there, and so is at least one hashed file outside them (the pin scripts). A test manifest in a shape the
// real one does not have passed every other test here while the release row could not read the real file.
func TestParityRecord_readsTheCommittedManifest(t *testing.T) {
	root, err := repoRoot()
	if err != nil {
		t.Fatal(err)
	}
	b, err := paritykey.ManifestJSON(paritykey.Dir(root))
	if err != nil {
		t.Fatal(err)
	}
	paths, err := parityQualifyingPaths(b)
	if err != nil {
		t.Fatalf("the committed manifest does not parse: %v", err)
	}
	outside := 0
	for _, p := range paths {
		if _, err := os.Stat(filepath.Join(root, p)); err != nil {
			t.Errorf("qualifying path %s is not in the repository", p)
		}
		if !slices.Contains(parityQualifyingTrees, p) {
			outside++
		}
	}
	for _, d := range parityQualifyingTrees {
		if !slices.Contains(paths, d) {
			t.Errorf("the qualifying paths lack %s/", d)
		}
	}
	if outside == 0 {
		t.Error("no hashed file outside the four directories: the committed manifest hashes scripts, and they were lost")
	}
	// And the row itself, on this repository: whatever it reads, it must not be a failure to read the manifest.
	if r := rowParity(liveRelEnv(relEnv{root: root})); strings.Contains(strings.Join(r.Detail, " "), parityManifestPath+":") {
		t.Errorf("the release row cannot read the committed manifest: %+v", r)
	}
}

// A sweep covers a later commit that changed docs or an unhashed script, and does not cover one that changed any of the four
// directories or a hashed file outside them.
func TestParityRecord_coversLaterCommitsOnlyIfTheHashedCodeIsUnchanged(t *testing.T) {
	dir, commit := parityTestRepo(t)
	ran := gitIn(t, dir, "rev-parse", "HEAD")
	paths, objOf := parityPathsIn(t, dir), objOfIn(dir)
	recs := []parityRecord{prec(ran, "2026-10-11T10:00:00Z", "PASS")}

	if r, why := parityCoverage(nil, ran, paths, objOf); r != nil || why != "no record" {
		t.Errorf("no record: covered=%v why=%q", r != nil, why)
	}
	if r, why := parityCoverage(recs, ran, paths, objOf); r == nil {
		t.Fatalf("a complete sweep at HEAD does not cover HEAD: %s", why)
	}
	for _, f := range []string{"docs/f.txt", "scripts/unrelated.py"} {
		gitIn(t, dir, "checkout", "-q", "-B", "side", ran)
		head := commit(f, "edit")
		if r, why := parityCoverage(recs, head, paths, objOf); r == nil {
			t.Errorf("a change to %s voided the sweep: %s", f, why)
		}
	}
	for _, f := range []string{"decoder/f.txt", "cuda/f.txt", "metal/f.txt", "gpu/f.txt", "scripts/pin.py", "scripts/own.py"} {
		gitIn(t, dir, "checkout", "-q", "-B", "side", ran) // each change on its own, from the commit the sweep was made at
		head := commit(f, "edit")
		name := f
		if i := strings.Index(f, "/f.txt"); i > 0 {
			name = f[:i] + "/"
		}
		r, why := parityCoverage(recs, head, paths, objOf)
		if r != nil || !strings.Contains(why, name+" changed since") {
			t.Errorf("a change to %s left the sweep covering the commit (covered=%v, %q)", f, r != nil, why)
		}
	}
}

// A file the manifest hashes today that did not exist when the sweep ran has changed since it.
func TestParityRecord_aHashedFileAddedSinceTheSweepVoidsIt(t *testing.T) {
	dir, commit := parityTestRepo(t)
	ran := gitIn(t, dir, "rev-parse", "HEAD")
	commit("scripts/new.py", "new")
	writeTree(t, dir, parityRecordsTree(t, strings.Replace(parityTestManifest, `"scripts/pin.py"]`, `"scripts/pin.py", "scripts/new.py"]`, 1)))
	gitIn(t, dir, "add", ".")
	gitIn(t, dir, "commit", "-q", "-m", "the manifest hashes a new file")
	head := gitIn(t, dir, "rev-parse", "HEAD")
	r, why := parityCoverage([]parityRecord{prec(ran, "2026-10-11T10:00:00Z", "PASS")}, head, parityPathsIn(t, dir), objOfIn(dir))
	if r != nil || !strings.Contains(why, "scripts/new.py changed since") {
		t.Errorf("a newly hashed file left the sweep covering the commit (covered=%v, %q)", r != nil, why)
	}
}

// Only a sweep with no blocker, on a clean tree, with the real-checkpoint tier run and no GATE_RUN narrowing counts; a run that
// is none of those at the same code is named, not skipped.
func TestParityRecord_onlyACompleteSweepCounts(t *testing.T) {
	dir, _ := parityTestRepo(t)
	head := gitIn(t, dir, "rev-parse", "HEAD")
	paths, objOf := parityPathsIn(t, dir), objOfIn(dir)
	for name, mod := range map[string]func(*parityRecord){
		"blockers":    func(r *parityRecord) { r.Verdict = "BLOCKERS"; r.Blockers = 2 },
		"dirty":       func(r *parityRecord) { r.Dirty = true },
		"no realckpt": func(r *parityRecord) { r.Realckpt = false },
		"scoped":      func(r *parityRecord) { r.Scoped = "Qwen" },
	} {
		r := prec(head, "2026-10-11T10:00:00Z", "PASS")
		mod(&r)
		got, why := parityCoverage([]parityRecord{r}, head, paths, objOf)
		if got != nil || !strings.Contains(why, "does not count") {
			t.Errorf("%s: covered=%v why=%q", name, got != nil, why)
		}
	}
}

// A later sweep with blockers at unchanged code is not hidden by an earlier clean one: the newest record at the same code decides.
func TestParityRecord_theNewestRecordAtTheSameCodeDecides(t *testing.T) {
	dir, commit := parityTestRepo(t)
	a := gitIn(t, dir, "rev-parse", "HEAD")
	b := commit("docs/f.txt", "docs")
	paths, objOf := parityPathsIn(t, dir), objOfIn(dir)
	bad := prec(b, "2026-10-11T11:00:00Z", "BLOCKERS")
	recs := []parityRecord{prec(a, "2026-10-11T10:00:00Z", "PASS"), bad}
	if r, why := parityCoverage(recs, b, paths, objOf); r != nil || !strings.Contains(why, "verdict BLOCKERS") {
		t.Errorf("an earlier clean sweep hid a later one with blockers at the same code: covered=%v %q", r != nil, why)
	}
	c := commit("decoder/f.txt", "code")
	if r, why := parityCoverage(recs, c, paths, objOf); r != nil || !strings.Contains(why, "decoder/ changed since") {
		t.Errorf("records at older code cover a later commit: covered=%v %q", r != nil, why)
	}
}

// The release row: OWED with no record (naming the command, the box and the record path), done with a complete sweep at the
// same code, OWED again once hashed code changes.
func TestRowParity(t *testing.T) {
	dir, commit := parityTestRepo(t)
	head := gitIn(t, dir, "rev-parse", "HEAD")
	e := liveRelEnv(relEnv{root: dir})
	r := rowParity(e)
	all := r.Record + " " + strings.Join(r.Detail, " ")
	if r.State != "OWED" {
		t.Fatalf("no record read as covered: %+v", r)
	}
	for _, want := range []string{"no record", "go run ./cmd/gate parity -logdir", "nobara", "night", parityRecordDirDefault + "/parity-<time>-<commit>.json"} {
		if !strings.Contains(all, want) {
			t.Errorf("the owed row does not say %q: %+v", want, r)
		}
	}
	b, _ := json.Marshal(prec(head, "2026-10-11T10:00:00Z", "PASS"))
	writeTree(t, dir, map[string]string{parityRecordDirDefault + "/parity-x.json": string(b)})
	commit("docs/f.txt", "doc only")
	if r := rowParity(e); r.State != "done" || !strings.Contains(strings.Join(r.Detail, " "), head[:8]) {
		t.Errorf("a complete sweep at the same code (a docs commit since): %+v", r)
	}
	commit("scripts/pin.py", "the pin script changed")
	if r := rowParity(e); r.State != "OWED" || !strings.Contains(strings.Join(r.Detail, " "), "scripts/pin.py changed since") {
		t.Errorf("a hashed file changed since the sweep: %+v", r)
	}
	// A file in the record directory that is not a record must not read as "no record".
	writeTree(t, dir, map[string]string{parityRecordDirDefault + "/parity-broken.json": "{"})
	if r := rowParity(e); r.State != "OWED" || !strings.Contains(strings.Join(r.Detail, " "), "not a parity-sweep record") {
		t.Errorf("an unreadable record: %+v", r)
	}
}

// `gate parity` writes a record the release row can read back: the run's commit, verdict and what it ran.
func TestParityRecord_writtenAndReadBack(t *testing.T) {
	dir, _ := parityTestRepo(t)
	head := gitIn(t, dir, "rev-parse", "HEAD")
	var out bytes.Buffer
	cells := []cellResult{{Pass: 40, Skip: 3}, {Pass: 12, Fail: 1, RC: 1}}
	writeParityRecord(&out, dir, parityRecordDirDefault, parityRun{Realckpt: true, Blockers: 1, Gaps: 2, FirstRuns: 0, Required: 55, Cells: cells})
	recs, err := loadParityRecords(filepath.Join(dir, parityRecordDirDefault))
	if err != nil || len(recs) != 1 {
		t.Fatalf("%d records read back (%v); output: %s", len(recs), err, out.String())
	}
	r := recs[0]
	if r.Commit != head || r.Verdict != "BLOCKERS" || r.Blockers != 1 || r.Gaps != 2 || !r.Realckpt || r.Required != 55 || len(r.Cells) != 2 || r.Cells[1].Fail != 1 {
		t.Errorf("the record does not carry the run: %+v", r)
	}
	if r.Dirty {
		t.Error("a run on a clean tree was recorded as dirty")
	}
	if len(r.Paths) == 0 || r.Paths["decoder"] == "" || r.Paths["scripts/pin.py"] == "" {
		t.Errorf("the record does not list the qualifying paths' objects: %v", r.Paths)
	}
	if !strings.Contains(out.String(), "record: ") {
		t.Errorf("the run did not say where its record went: %q", out.String())
	}
	// An empty record directory writes nothing; a sweep with no blocker writes PASS.
	writeParityRecord(&out, dir, "", parityRun{Realckpt: true})
	if recs2, _ := loadParityRecords(filepath.Join(dir, parityRecordDirDefault)); len(recs2) != 1 {
		t.Errorf("an empty record directory still wrote a record (%d files)", len(recs2))
	}
	writeParityRecord(&out, dir, "elsewhere", parityRun{Realckpt: true, Scoped: "Qwen"})
	if recs3, err := loadParityRecords(filepath.Join(dir, "elsewhere")); err != nil || len(recs3) != 1 || recs3[0].Verdict != "PASS" || recs3[0].Scoped != "Qwen" {
		t.Errorf("a sweep with no blocker, narrowed by GATE_RUN: %+v (%v)", recs3, err)
	}
}

// The release command reads twelve rows, the parity sweep among them.
func TestRelease_rowsIncludeTheParitySweep(t *testing.T) {
	if len(releaseRows) != 12 {
		t.Errorf("gate release has %d rows, want 12", len(releaseRows))
	}
	found := false
	for _, f := range releaseRows {
		if reflect.ValueOf(f).Pointer() == reflect.ValueOf(rowParity).Pointer() {
			found = true
		}
	}
	if !found {
		t.Error("no release row is the parity sweep's")
	}
}

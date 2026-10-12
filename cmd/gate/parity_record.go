package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// The parity sweep's record. `gate parity` prints a verdict and keeps its raw streams in the log directory; RELEASING.md's
// §C1 asks for a sweep before a tag, and nothing durable said "this commit, swept, no blocker". Every run now writes one small
// JSON file, committed beside the other measurement records, and `gate release` reads them: the sweep covers a commit when the
// newest record made at the same code is a complete sweep with no blocker. "The same code" is every path in
// parityQualifyingPaths: the files the parity manifest hashes, and the four directories a GPU-gate run vouches for.

// parityRecordDirDefault is where a sweep's record goes, relative to the repository root.
const parityRecordDirDefault = "docs/measurements/parity-sweep"

// parityManifestPath is the parity manifest, relative to the repository root.
const parityManifestPath = "testdata/parity_manifest.json"

// parityQualifyingTrees are the directories whose every file a sweep vouches for, hashed by the manifest or not.
var parityQualifyingTrees = []string{"decoder", "cuda", "metal", "gpu"}

// parityRecordCell is one `go test` cell of the sweep.
type parityRecordCell struct {
	Name string `json:"name"`
	Pass int    `json:"pass"`
	Skip int    `json:"skip"`
	Fail int    `json:"fail"`
	RC   int    `json:"rc"`
}

// parityRecord is one run of `gate parity`, whatever its verdict. Only a complete sweep with no blocker can qualify (valid).
type parityRecord struct {
	Schema    int                `json:"schema"`
	Commit    string             `json:"commit"` // the full SHA the sweep was made at
	Date      string             `json:"date"`   // UTC, RFC 3339
	Host      string             `json:"host"`
	Verdict   string             `json:"verdict"` // PASS, or BLOCKERS
	Dirty     bool               `json:"dirty"`
	Realckpt  bool               `json:"realckpt"`         // the real-checkpoint tier ran (REALCKPT=1)
	Scoped    string             `json:"scoped,omitempty"` // GATE_RUN's pattern, when the checkset was narrowed
	Required  int                `json:"required"`         // named gates in the checkset
	Blockers  int                `json:"blockers"`
	Gaps      int                `json:"gaps"`       // required gates whose asset has never been built (reported, not blockers)
	FirstRuns int                `json:"first_runs"` // failures with no confirmed prior result (reported, not blockers)
	Cells     []parityRecordCell `json:"cells"`
	Paths     map[string]string  `json:"paths"` // git object id of each qualifying path at Commit, for a reader; qualification recomputes them
}

// parityRun is what runParity hands the record writer.
type parityRun struct {
	Realckpt                            bool
	Scoped                              string
	Required, Blockers, Gaps, FirstRuns int
	Cells                               []cellResult
}

// parityQualifyingPaths is the paths a sweep vouches for, sorted: parityQualifyingTrees, and every file the manifest hashes
// (its shared sets and each family's own list) that is not under one of them. A manifest that hashes no file is an error: the
// row would otherwise cover less than it says.
func parityQualifyingPaths(manifest []byte) ([]string, error) {
	var m struct {
		SharedSets map[string][]string  `json:"shared_sets"`
		Families   [][2]json.RawMessage `json:"families"`
	}
	if err := json.Unmarshal(manifest, &m); err != nil {
		return nil, fmt.Errorf("%s: %v", parityManifestPath, err)
	}
	set := map[string]bool{}
	hashed := 0
	add := func(files []string) {
		for _, f := range files {
			hashed++
			f = filepath.ToSlash(filepath.Clean(f))
			for _, d := range parityQualifyingTrees {
				if strings.HasPrefix(f, d+"/") {
					f = ""
					break
				}
			}
			if f != "" {
				set[f] = true
			}
		}
	}
	for _, files := range m.SharedSets {
		add(files)
	}
	for _, fam := range m.Families {
		var row struct {
			Own []string `json:"own"`
		}
		if err := json.Unmarshal(fam[1], &row); err != nil {
			return nil, fmt.Errorf("%s: a family row: %v", parityManifestPath, err)
		}
		add(row.Own)
	}
	if hashed == 0 {
		return nil, fmt.Errorf("%s hashes no file", parityManifestPath)
	}
	for _, d := range parityQualifyingTrees {
		set[d] = true
	}
	paths := make([]string, 0, len(set))
	for p := range set {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return paths, nil
}

// gitObject is the git object id of path (a file or a directory) at rev, and "" when rev has no such path. An unknown rev is an error.
func gitObject(root, rev, path string) (string, error) {
	out, err := gitOut(root, "ls-tree", rev, "--", path)
	if err != nil {
		return "", err
	}
	f := strings.Fields(out) // <mode> <type> <object>\t<path>
	if len(f) < 3 {
		return "", nil
	}
	return f[2], nil
}

// valid says whether the run itself is one a release can lean on: no blocker, on a clean tree, with the real-checkpoint tier
// run and the checkset not narrowed.
func (r parityRecord) valid() (bool, string) {
	switch {
	case r.Verdict != "PASS":
		return false, "verdict " + r.Verdict
	case r.Dirty:
		return false, "the sweep was made on a dirty tree"
	case !r.Realckpt:
		return false, "the real-checkpoint tier was not run (REALCKPT=0)"
	case r.Scoped != "":
		return false, "the checkset was narrowed (GATE_RUN=" + r.Scoped + ")"
	}
	return true, ""
}

// changedSince names the first qualifying path whose object at head differs from its object at the sweep's commit (a directory
// with a trailing slash), or "". A path one side lacks has changed. An error reading either side is returned, not read as
// "unchanged".
func (r parityRecord) changedSince(head string, paths []string, objOf func(rev, path string) (string, error)) (string, error) {
	for _, p := range paths {
		then, err1 := objOf(r.Commit, p)
		now, err2 := objOf(head, p)
		if err1 != nil || err2 != nil {
			return "", fmt.Errorf("could not compare %s (%v %v)", p, err1, err2)
		}
		if then != now {
			for _, d := range parityQualifyingTrees {
				if p == d {
					return p + "/", nil
				}
			}
			return p, nil
		}
	}
	return "", nil
}

// parityCoverage is whether the records cover head. The newest record made at the same code decides, whatever its verdict: a
// later sweep with blockers at unchanged code is not hidden by an earlier clean one. With no record at the same code, the reason
// names the newest record and what changed since it.
func parityCoverage(recs []parityRecord, head string, paths []string, objOf func(rev, path string) (string, error)) (*parityRecord, string) {
	newestWhy := ""
	for i := len(recs) - 1; i >= 0; i-- {
		r := &recs[i]
		changed, err := r.changedSince(head, paths, objOf)
		if err != nil {
			return nil, err.Error()
		}
		if changed == "" {
			if ok, why := r.valid(); !ok {
				return nil, fmt.Sprintf("the newest record at this code (%s @ %s) does not count: %s", r.Date, short8(r.Commit), why)
			}
			return r, ""
		}
		if newestWhy == "" {
			newestWhy = fmt.Sprintf("newest record %s @ %s: %s changed since", r.Date, short8(r.Commit), changed)
		}
	}
	if newestWhy == "" {
		return nil, "no record"
	}
	return nil, newestWhy
}

func short8(sha string) string {
	if len(sha) > 8 {
		return sha[:8]
	}
	return sha
}

// loadParityRecords reads every record in dir, oldest first. A file that is not a record is an error, not a skip: a record that
// cannot be read must not read as "no record".
func loadParityRecords(dir string) ([]parityRecord, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var recs []parityRecord
	for _, e := range ents {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		var r parityRecord
		if err := json.Unmarshal(b, &r); err != nil || r.Schema != 1 || r.Commit == "" || r.Verdict == "" {
			return nil, fmt.Errorf("%s is not a parity-sweep record (schema 1 with a commit and a verdict): %v", e.Name(), err)
		}
		recs = append(recs, r)
	}
	sort.SliceStable(recs, func(i, j int) bool { return recs[i].Date < recs[j].Date })
	return recs, nil
}

// writeParityRecord writes one sweep's record under recordDir (relative to root unless absolute; empty writes none). A record
// that cannot be written is said loudly and does not change the sweep's exit code: the release command reads a missing record
// as owed.
func writeParityRecord(w io.Writer, root, recordDir string, run parityRun) {
	if recordDir == "" {
		return
	}
	fail := func(err error) { fmt.Fprintf(w, "\n  RECORD NOT WRITTEN: %v\n", err) }
	dir := recordDir
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(root, dir)
	}
	full, err := gitOut(root, "rev-parse", "HEAD")
	if err != nil {
		fail(err)
		return
	}
	status, err := gitOut(root, "status", "--porcelain")
	if err != nil {
		fail(err)
		return
	}
	now := time.Now().UTC()
	rec := parityRecord{
		Schema: 1, Commit: full, Date: now.Format(time.RFC3339), Verdict: "PASS", Dirty: status != "",
		Realckpt: run.Realckpt, Scoped: run.Scoped, Required: run.Required,
		Blockers: run.Blockers, Gaps: run.Gaps, FirstRuns: run.FirstRuns, Paths: map[string]string{},
	}
	if run.Blockers > 0 {
		rec.Verdict = "BLOCKERS"
	}
	if b, err := exec.Command("uname", "-sm").Output(); err == nil {
		rec.Host = strings.TrimSpace(string(b))
	}
	for _, c := range run.Cells {
		rec.Cells = append(rec.Cells, parityRecordCell{Name: c.Cell.Name, Pass: c.Pass, Skip: c.Skip, Fail: c.Fail, RC: c.RC})
	}
	if mb, err := os.ReadFile(filepath.Join(root, parityManifestPath)); err == nil {
		if paths, err := parityQualifyingPaths(mb); err == nil {
			for _, p := range paths {
				if id, err := gitObject(root, full, p); err == nil && id != "" {
					rec.Paths[p] = id
				}
			}
		}
	}
	b, err := json.MarshalIndent(rec, "", " ")
	if err != nil {
		fail(err)
		return
	}
	path := filepath.Join(dir, fmt.Sprintf("parity-%s-%s.json", now.Format("20060102T150405Z"), full[:8]))
	if err := writeFileAtomic(path, append(b, '\n')); err != nil {
		fail(err)
		return
	}
	fmt.Fprintf(w, "\n  record: %s (commit it with the release; `gate release` reads it)\n", path)
}

// rowParity is the release command's §C1 row: a complete parity sweep covers HEAD.
func rowParity(e *relEnv) relRow {
	const name = "the §C1 parity sweep covers HEAD"
	const want = parityRecordDirDefault + "/parity-<time>-<commit>.json"
	const how = "run `go run ./cmd/gate parity -logdir <durable dir>` on nobara, at night (it takes hours), and commit the record it writes"
	recs, err := loadParityRecords(filepath.Join(e.root, parityRecordDirDefault))
	if err != nil {
		return owed(name, want, err.Error())
	}
	mb, err := os.ReadFile(filepath.Join(e.root, parityManifestPath))
	if err != nil {
		return owed(name, want, err.Error())
	}
	paths, err := parityQualifyingPaths(mb)
	if err != nil {
		return owed(name, want, err.Error())
	}
	head, err := gitOut(e.root, "rev-parse", "HEAD")
	if err != nil {
		return owed(name, want, err.Error())
	}
	r, why := parityCoverage(recs, head, paths, func(rev, path string) (string, error) { return gitObject(e.root, rev, path) })
	if r == nil {
		return owed(name, want, why, how)
	}
	detail := fmt.Sprintf("%s @ %s on %s: %d required gates, no blocker", r.Date, short8(r.Commit), r.Host, r.Required)
	if r.Gaps > 0 || r.FirstRuns > 0 {
		detail += fmt.Sprintf(" (%d coverage gap(s) and %d first-run failure(s) reported, as the sweep prints them)", r.Gaps, r.FirstRuns)
	}
	return done(name, parityRecordDirDefault, detail)
}

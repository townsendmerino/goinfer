package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// The record of a `gate quick -full` run. By day quick defers the real-checkpoint cells and an untouched device
// module's long kernel tests (deferCells), so before a tag something durable has to say they ran: one small JSON file
// per run, committed beside the other measurement records. `gate release` reads them (rowQuickFull): an operating
// system is covered at a commit when the newest record made on it at the same code is a PASS on a clean tree.

// quickRecordDirDefault is where a -full run's record goes, relative to the repository root.
const quickRecordDirDefault = "docs/measurements/quick-full"

// quickRecordSystems are the operating systems a release needs a record from: darwin builds and runs the metal
// module, linux the cuda one, and neither box can run the other's.
var quickRecordSystems = []string{"darwin", "linux"}

// quickRecord is one `gate quick -full` run, whatever its verdict.
type quickRecord struct {
	Schema  int      `json:"schema"`
	OS      string   `json:"os"`
	Arch    string   `json:"arch"`
	Commit  string   `json:"commit"` // the full SHA the run was made at
	Date    string   `json:"date"`   // UTC, RFC 3339
	Verdict string   `json:"verdict"`
	Dirty   bool     `json:"dirty"`
	Full    bool     `json:"full"`
	Pass    int      `json:"pass"`
	Skip    int      `json:"skip"`
	Fail    int      `json:"fail"`
	Cells   int      `json:"cells"`
	Ran     int      `json:"ran"`
	Cached  int      `json:"cached"`
	NotRun  []string `json:"not_run,omitempty"` // cells of modules that do not build on this system
}

// quickSummary is what reportQuick counted, for the record.
type quickSummary struct {
	Pass, Skip, Fail   int
	Cells, Ran, Cached int
	NotRun             []string
}

// writeQuickRecord writes a -full run's record into dir (relative to root unless absolute). A record that cannot be
// written is said loudly and does not change the exit code: the release command reads the missing record as owed.
func writeQuickRecord(w io.Writer, root, dir string, dirty bool, rc int, sum quickSummary) {
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(root, dir)
	}
	full, err := gitOut(root, "rev-parse", "HEAD")
	if err != nil {
		fmt.Fprintf(w, "\n  RECORD NOT WRITTEN: %v\n", err)
		return
	}
	verdict := "PASS"
	if rc != 0 {
		verdict = "FAIL"
	}
	now := time.Now().UTC()
	rec := quickRecord{Schema: 1, OS: runtime.GOOS, Arch: runtime.GOARCH, Commit: full, Date: now.Format(time.RFC3339),
		Verdict: verdict, Dirty: dirty, Full: true, Pass: sum.Pass, Skip: sum.Skip, Fail: sum.Fail,
		Cells: sum.Cells, Ran: sum.Ran, Cached: sum.Cached, NotRun: sum.NotRun}
	b, err := json.MarshalIndent(rec, "", " ")
	if err != nil {
		fmt.Fprintf(w, "\n  RECORD NOT WRITTEN: %v\n", err)
		return
	}
	path := filepath.Join(dir, fmt.Sprintf("%s-%s-%s-%s.json", rec.OS, rec.Arch, now.Format("20060102T150405Z"), full[:8]))
	if err := writeFileAtomic(path, append(b, '\n')); err != nil {
		fmt.Fprintf(w, "\n  RECORD NOT WRITTEN: %v\n", err)
		return
	}
	fmt.Fprintf(w, "\n  record: %s (commit it with the release; `gate release` reads it)\n", path)
}

// loadQuickRecords reads every record in dir, oldest first. A file that is not a record is an error, not a skip.
func loadQuickRecords(dir string) ([]quickRecord, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var recs []quickRecord
	for _, e := range ents {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		var r quickRecord
		if err := json.Unmarshal(b, &r); err != nil || r.Schema != 1 || r.OS == "" || r.Commit == "" {
			return nil, fmt.Errorf("%s is not a quick -full record (schema 1 with an os and a commit): %v", e.Name(), err)
		}
		recs = append(recs, r)
	}
	sort.SliceStable(recs, func(i, j int) bool { return recs[i].Date < recs[j].Date })
	return recs, nil
}

// valid says whether the run is one a release can lean on: every cell run, green, on a clean tree.
func (r quickRecord) valid() (bool, string) {
	switch {
	case !r.Full:
		return false, "the run deferred cells (not -full)"
	case r.Verdict != "PASS":
		return false, "verdict " + r.Verdict
	case r.Dirty:
		return false, "the run was made on a dirty tree"
	case r.Pass == 0:
		return false, "no test passed"
	}
	return true, ""
}

// quickRecordVoidedBy reports whether a change to path, since a run, voids the run. Everything a test can compile or
// read does; the documentation tree, the website, the CI definitions and the top-level prose files do not, so a
// release commit that edits the CHANGELOG and commits the records keeps its run.
func quickRecordVoidedBy(path string) bool {
	for _, p := range []string{"docs/", "site/", ".github/"} {
		if strings.HasPrefix(path, p) {
			return false
		}
	}
	return strings.Contains(path, "/") || !strings.HasSuffix(path, ".md")
}

// quickCoverage is, for one operating system, whether the records cover head. The newest record made there at the
// same code decides, whatever its verdict, so a later FAIL at unchanged code is not hidden by an earlier PASS.
// changed lists the paths that differ between two revisions.
func quickCoverage(recs []quickRecord, goos, head string, changed func(from, to string) ([]string, error)) (*quickRecord, string) {
	var newestWhy string
	for i := len(recs) - 1; i >= 0; i-- {
		r := &recs[i]
		if r.OS != goos {
			continue
		}
		paths, err := changed(r.Commit, head)
		if err != nil {
			return nil, fmt.Sprintf("could not compare %s with HEAD: %v", r.Commit[:8], err)
		}
		first := ""
		for _, p := range paths {
			if quickRecordVoidedBy(p) {
				first = p
				break
			}
		}
		if first == "" {
			if ok, why := r.valid(); !ok {
				return nil, fmt.Sprintf("the newest record at this code (%s @ %s) does not count: %s", r.Date, r.Commit[:8], why)
			}
			return r, ""
		}
		if newestWhy == "" {
			newestWhy = fmt.Sprintf("newest record %s @ %s: %s changed since", r.Date, r.Commit[:8], first)
		}
	}
	if newestWhy == "" {
		return nil, "no record"
	}
	return nil, newestWhy
}

// rowQuickFull is the release row for the tests the day loop defers: a covering `gate quick -full` record from each
// of quickRecordSystems.
func rowQuickFull(e *relEnv) relRow {
	const name = "the deferred tests (gate quick -full) on both systems cover HEAD"
	recs, err := loadQuickRecords(filepath.Join(e.root, quickRecordDirDefault))
	if err != nil {
		return owed(name, quickRecordDirDefault, err.Error())
	}
	head, err := gitOut(e.root, "rev-parse", "HEAD")
	if err != nil {
		return owed(name, quickRecordDirDefault, err.Error())
	}
	changed := func(from, to string) ([]string, error) {
		out, err := gitOut(e.root, "diff", "--name-only", from, to)
		if err != nil {
			return nil, err
		}
		if out == "" {
			return nil, nil
		}
		return strings.Split(out, "\n"), nil
	}
	var bad, ok []string
	for _, goos := range quickRecordSystems {
		if r, why := quickCoverage(recs, goos, head, changed); r != nil {
			ok = append(ok, fmt.Sprintf("%s: %s @ %s (%d pass, %d skip)", goos, r.Date, r.Commit[:8], r.Pass, r.Skip))
		} else {
			bad = append(bad, goos+": "+why)
		}
	}
	if len(bad) > 0 {
		return owed(name, quickRecordDirDefault+"/<os>-<arch>-<time>-<commit>.json",
			append(bad, "run `go run ./cmd/gate quick -full -base <the previous tag>` on that box (about 10 minutes on the Mac; queue it at night) and commit the record")...)
	}
	return done(name, quickRecordDirDefault, ok...)
}

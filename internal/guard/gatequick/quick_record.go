package gatequick

import (
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
)

// The record of a `gate quick -full` run. By day quick defers the real-checkpoint cells and an untouched device
// module's long kernel tests (deferCells), so before a tag something durable has to say they ran: one small JSON file
// per run, committed beside the other measurement records. `gate release` reads them (rowQuickFull): an operating
// system is covered at a commit when the newest record made on it at the same code is a PASS on a clean tree.

// QuickRecordDirDefault is where a -full run's record goes, relative to the repository root.
const QuickRecordDirDefault = "docs/measurements/quick-full"

// QuickRecord is one `gate quick -full` run, whatever its verdict.
type QuickRecord struct {
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

// QuickSummary is what reportQuick counted, for the record.
type QuickSummary struct {
	Pass, Skip, Fail   int
	Cells, Ran, Cached int
	NotRun             []string
}

// writeQuickRecord writes a -full run's record into dir (relative to root unless absolute). A record that cannot be
// written is said loudly and does not change the exit code: the release command reads the missing record as owed.
func writeQuickRecord(w io.Writer, root, dir string, dirty bool, rc int, sum QuickSummary) {
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(root, dir)
	}
	full, err := gatecore.GitOut(root, "rev-parse", "HEAD")
	if err != nil {
		fmt.Fprintf(w, "\n  RECORD NOT WRITTEN: %v\n", err)
		return
	}
	verdict := "PASS"
	if rc != 0 {
		verdict = "FAIL"
	}
	now := time.Now().UTC()
	rec := QuickRecord{Schema: 1, OS: runtime.GOOS, Arch: runtime.GOARCH, Commit: full, Date: now.Format(time.RFC3339),
		Verdict: verdict, Dirty: dirty, Full: true, Pass: sum.Pass, Skip: sum.Skip, Fail: sum.Fail,
		Cells: sum.Cells, Ran: sum.Ran, Cached: sum.Cached, NotRun: sum.NotRun}
	b, err := json.MarshalIndent(rec, "", " ")
	if err != nil {
		fmt.Fprintf(w, "\n  RECORD NOT WRITTEN: %v\n", err)
		return
	}
	path := filepath.Join(dir, fmt.Sprintf("%s-%s-%s-%s.json", rec.OS, rec.Arch, now.Format("20060102T150405Z"), full[:8]))
	if err := gatecore.WriteFileAtomic(path, append(b, '\n')); err != nil {
		fmt.Fprintf(w, "\n  RECORD NOT WRITTEN: %v\n", err)
		return
	}
	fmt.Fprintf(w, "\n  record: %s (commit it with the release; `gate release` reads it)\n", path)
}

// LoadQuickRecords reads every record in dir, oldest first. A file that is not a record is an error, not a skip.
func LoadQuickRecords(dir string) ([]QuickRecord, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var recs []QuickRecord
	for _, e := range ents {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		var r QuickRecord
		if err := json.Unmarshal(b, &r); err != nil || r.Schema != 1 || r.OS == "" || r.Commit == "" {
			return nil, fmt.Errorf("%s is not a quick -full record (schema 1 with an os and a commit): %v", e.Name(), err)
		}
		recs = append(recs, r)
	}
	slices.SortStableFunc(recs, func(a, b QuickRecord) int { return cmp.Compare(a.Date, b.Date) })
	return recs, nil
}

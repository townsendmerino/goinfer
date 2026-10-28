package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// identityRunRecord is what a `gate identity` run leaves for its grader (SB12, docs/tasks/task-subsystems-2026-10.md
// §8), in its run directory beside the dumps: what it printed before the verdict, the options and revisions the report
// names, each cell, why a run of a family ended badly, the selected families' manifest rows, the grade inputs, and how
// the -record write went. The dumps it judges stay in dumps/<run>/<family>/ and survive only with -keep, so `gate
// identity -replay FILE` refuses a record whose run did not keep them.
type identityRunRecord struct {
	Schema   int                       `json:"schema"`
	Prelude  string                    `json:"prelude"`
	Opts     identityRecordOpts        `json:"opts"`
	Old      identityRecordSide        `json:"old"`
	New      identityRecordSide        `json:"new"`
	Manifest map[string]manifestFamily `json:"manifest"`
	Selected []string                  `json:"selected"`
	Cells    []*identityCell           `json:"cells"`
	Crash    map[string]string         `json:"crash,omitempty"`
	Inputs   identityGradeInputs       `json:"inputs"`
	RowsErr  string                    `json:"rows_err,omitempty"`
	Exit     int                       `json:"exit"`
}

// identityRecordOpts is the part of identityOpts the grader reads.
type identityRecordOpts struct {
	Backend string `json:"backend"`
	Giw     string `json:"giw,omitempty"`
	Assets  string `json:"assets"`
	Record  string `json:"record,omitempty"`
	Keep    bool   `json:"keep"`
}

type identityRecordSide struct {
	SHA   string `json:"sha"`
	Short string `json:"short"`
}

const identityRunRecordFile = "identity_run.json"

// writeIdentityRunRecord writes the run's record into its run directory. A record that cannot be written is said loudly
// and does not change the exit code.
func writeIdentityRunRecord(w io.Writer, wd, prelude string, o identityOpts, old, nw *identitySide, manifest map[string]manifestFamily,
	selected []string, cells []*identityCell, crash map[string]string, in identityGradeInputs, rowsErr string, rc int) {
	sel := map[string]manifestFamily{}
	for _, f := range selected {
		sel[f] = manifest[f]
	}
	rec := identityRunRecord{Schema: 1, Prelude: prelude,
		Opts:     identityRecordOpts{Backend: o.Backend, Giw: o.Giw, Assets: o.Assets, Record: o.Record, Keep: o.Keep},
		Old:      identityRecordSide{SHA: old.SHA, Short: old.Short},
		New:      identityRecordSide{SHA: nw.SHA, Short: nw.Short},
		Manifest: sel, Selected: selected, Cells: cells, Crash: crash, Inputs: in, RowsErr: rowsErr, Exit: rc}
	b, err := json.MarshalIndent(rec, "", "  ")
	if err == nil {
		err = os.WriteFile(filepath.Join(wd, identityRunRecordFile), append(b, '\n'), 0o644)
	}
	if err != nil {
		fmt.Fprintf(w, "gate identity: the run record was not written: %v\n", err)
	}
}

// replayIdentity prints a recorded identity run's report as the run printed it: the prelude, then each cell judged
// again from its kept dumps and the report and verdict from gradeIdentity, with the -record write answered from the
// record.
func replayIdentity(w io.Writer, path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate identity: %v\n", err)
		return 2
	}
	var rec identityRunRecord
	if err := json.Unmarshal(b, &rec); err != nil || rec.Schema != 1 {
		fmt.Fprintf(os.Stderr, "gate identity: %s is not an identity run record (schema 1): %v\n", path, err)
		return 2
	}
	if !rec.Opts.Keep {
		fmt.Fprintf(os.Stderr, "gate identity: %s was recorded without -keep, so its dumps are gone and it cannot be regraded\n", path)
		return 2
	}
	wd := filepath.Dir(path)
	var outs []cellOutcome
	for _, c := range rec.Cells {
		if c.NotRun != "" {
			outs = append(outs, judgeCell(c, rec.Opts.Backend, nil, nil, nil))
			continue
		}
		dir := func(run string) string { return filepath.Join(wd, "dumps", run, c.Family) }
		outs = append(outs, judgeCell(c, rec.Opts.Backend, readDump(dir("new1"), c.ID, rec.Crash["new1/"+c.Family]),
			readDump(dir("new2"), c.ID, rec.Crash["new2/"+c.Family]), readDump(dir("old"), c.ID, rec.Crash["old/"+c.Family])))
	}
	o := identityOpts{Backend: rec.Opts.Backend, Giw: rec.Opts.Giw, Assets: rec.Opts.Assets, Record: rec.Opts.Record, Keep: rec.Opts.Keep}
	old := &identitySide{Label: "old", SHA: rec.Old.SHA, Short: rec.Old.Short}
	nw := &identitySide{Label: "new", SHA: rec.New.SHA, Short: rec.New.Short}
	writeRows := func([]string) error {
		if rec.RowsErr != "" {
			return fmt.Errorf("%s", rec.RowsErr)
		}
		return nil
	}
	fmt.Fprint(w, rec.Prelude)
	return gradeIdentity(w, o, old, nw, rec.Manifest, rec.Selected, outs, rec.Inputs, writeRows)
}

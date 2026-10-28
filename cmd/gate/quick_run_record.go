package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// quickRunRecord is what a `gate quick` run leaves for its grader (SB12, docs/tasks/task-subsystems-2026-10.md §8):
// what quick printed before the verdict, each job's outcome inputs (a test job's stream, a lint job's output), the
// cells it could not run or deferred, the number of goldens the selection triggered, and the run's start and end.
// `gate quick -replay FILE` prints the report exactly as the run did.
type quickRunRecord struct {
	Schema   int               `json:"schema"`
	Prelude  string            `json:"prelude"`
	Verbose  bool              `json:"verbose,omitempty"`
	Goldens  int               `json:"goldens"`
	Start    time.Time         `json:"start"`
	End      time.Time         `json:"end"`
	Jobs     []quickRecordJob  `json:"jobs"`
	NotRun   []quickRecordCell `json:"not_run,omitempty"`
	Deferred []quickRecordCell `json:"deferred,omitempty"`
}

// quickRecordJob is one finished job. Stream is a test job's go test -json file, relative to the record; a lint job's
// verdict reads Out instead.
type quickRecordJob struct {
	Name    string   `json:"name"`
	Kind    string   `json:"kind"`
	Detail  string   `json:"detail,omitempty"`
	Mirrors string   `json:"mirrors,omitempty"`
	Canary  string   `json:"canary,omitempty"`
	QuietOK bool     `json:"quiet_ok,omitempty"`
	Out     string   `json:"out,omitempty"`
	Err     string   `json:"err,omitempty"`
	RC      int      `json:"rc"`
	LogPath string   `json:"log_path,omitempty"`
	Stream  string   `json:"stream,omitempty"`
	DurNS   int64    `json:"dur_ns"`
	Tier    string   `json:"tier,omitempty"`
	Golden  bool     `json:"golden,omitempty"`
	LogTail []string `json:"log_tail,omitempty"`
}

type quickRecordCell struct {
	Name     string `json:"name"`
	NotRun   string `json:"not_run,omitempty"`
	Deferred string `json:"deferred,omitempty"`
	Run      string `json:"run,omitempty"`
}

const quickRunRecordFile = "quick_run.json"

// writeQuickRunRecord writes the run's record into its run directory. A record that cannot be written is said loudly
// and does not change the exit code.
func writeQuickRunRecord(w io.Writer, rundir, prelude string, jobs []*quickJob, notRun, deferred []*quickCell, goldens int, t0, end time.Time, verbose bool) {
	if rundir == "" {
		return
	}
	rec := quickRunRecord{Schema: 1, Prelude: prelude, Verbose: verbose, Goldens: goldens, Start: t0, End: end}
	for _, jb := range jobs {
		rj := quickRecordJob{Name: jb.Name, Kind: jb.Kind, Detail: jb.Detail, Mirrors: jb.Mirrors, Canary: jb.Canary,
			QuietOK: jb.QuietOK, Out: jb.Out, RC: jb.RC, LogPath: jb.LogPath, DurNS: int64(jb.dur()), LogTail: jb.LogTail}
		if jb.Err != nil {
			rj.Err = jb.Err.Error()
		}
		if jb.JSONPath != "" {
			rj.Stream = filepath.Base(jb.JSONPath)
		}
		if jb.Cell != nil {
			rj.Tier, rj.Golden = jb.Cell.Tier, jb.Cell.Golden
		}
		rec.Jobs = append(rec.Jobs, rj)
	}
	for _, c := range notRun {
		rec.NotRun = append(rec.NotRun, quickRecordCell{Name: c.Name, NotRun: c.NotRun})
	}
	for _, c := range deferred {
		rec.Deferred = append(rec.Deferred, quickRecordCell{Name: c.Name, Deferred: c.Deferred, Run: c.Run})
	}
	b, err := json.MarshalIndent(rec, "", "  ")
	if err == nil {
		err = os.WriteFile(filepath.Join(rundir, quickRunRecordFile), append(b, '\n'), 0o644)
	}
	if err != nil {
		fmt.Fprintf(w, "gate quick: the run record was not written: %v\n", err)
	}
}

// replayQuick prints a recorded quick run's report as the run printed it: the prelude, then gradeQuick on the jobs
// rebuilt from the record, each test job's outcome derived again from its stream by gradeQuickTest.
func replayQuick(w io.Writer, path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate quick: %v\n", err)
		return 2
	}
	var rec quickRunRecord
	if err := json.Unmarshal(b, &rec); err != nil || rec.Schema != 1 {
		fmt.Fprintf(os.Stderr, "gate quick: %s is not a quick run record (schema 1): %v\n", path, err)
		return 2
	}
	var jobs []*quickJob
	for _, rj := range rec.Jobs {
		jb := &quickJob{Name: rj.Name, Kind: rj.Kind, Detail: rj.Detail, Mirrors: rj.Mirrors, Canary: rj.Canary,
			QuietOK: rj.QuietOK, Out: rj.Out, RC: rj.RC, LogPath: rj.LogPath, LogTail: rj.LogTail,
			Cell: &quickCell{Name: rj.Name, Tier: rj.Tier, Golden: rj.Golden}}
		jb.end = jb.start.Add(time.Duration(rj.DurNS))
		if rj.Err != "" {
			jb.Err = fmt.Errorf("%s", rj.Err)
		}
		if rj.Kind == "test" && rj.Stream != "" {
			f, err := os.Open(filepath.Join(filepath.Dir(path), rj.Stream))
			if err != nil {
				fmt.Fprintf(os.Stderr, "gate quick: %v\n", err)
				return 2
			}
			jb.JSONPath = rj.Stream
			jb.res = newResults()
			jb.res.cur = jb.Name
			jb.res.consume(f)
			f.Close()
			if jb.Err == nil {
				gradeQuickTest(jb)
			}
		}
		jobs = append(jobs, jb)
	}
	var notRun, deferred []*quickCell
	for _, c := range rec.NotRun {
		notRun = append(notRun, &quickCell{Name: c.Name, NotRun: c.NotRun})
	}
	for _, c := range rec.Deferred {
		deferred = append(deferred, &quickCell{Name: c.Name, Deferred: c.Deferred, Run: c.Run})
	}
	fmt.Fprint(w, rec.Prelude)
	return gradeQuick(w, jobs, notRun, deferred, rec.Goldens, rec.Start, rec.End, rec.Verbose, nil)
}

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// runRecord is what a tally gate's run leaves for its grader (SB12, docs/tasks/task-subsystems-2026-10.md §8): the
// configuration the verdict reads, the provenance the report prints, and each cell with its exit code and the file
// its `go test -json` stream was written to. `gate census|heavy -replay FILE` grades a record exactly as the run that
// wrote it did; TestGraderGoldens holds that on recorded runs.
type runRecord struct {
	Schema           int          `json:"schema"`
	Gate             string       `json:"gate"`
	Decision         string       `json:"decision"`
	TopLevelOnly     bool         `json:"top_level_only"`
	ZeroPolicy       string       `json:"zero_policy"`
	RCIsFailure      bool         `json:"rc_is_failure"`
	PkgFailIsFailure bool         `json:"pkg_fail_is_failure"`
	RequireFixtures  bool         `json:"require_fixtures,omitempty"`
	Provenance       provenance   `json:"provenance"`
	Cells            []recordCell `json:"cells"`
}

// recordCell is one cell of a run. Stream is the stream's file, relative to the record; LogPath is where the run
// wrote it, which the report names. Err is set when the cell never started, and then there is no stream.
type recordCell struct {
	Name    string   `json:"name"`
	Pkgs    []string `json:"pkgs,omitempty"`
	Tags    []string `json:"tags,omitempty"`
	Run     string   `json:"run,omitempty"`
	RC      int      `json:"rc"`
	Err     string   `json:"err,omitempty"`
	Stream  string   `json:"stream,omitempty"`
	LogPath string   `json:"log_path,omitempty"`
	DurNS   int64    `json:"dur_ns"`
}

const runRecordSchema = 1

// runRecordPath is where a gate's run writes its record: beside the cells' streams in the log directory.
func runRecordPath(logDir string, cfg *gateConfig) string {
	return filepath.Join(logDir, "gate_"+sanitize(cfg.Name)+"_run.json")
}

// writeRunRecord writes the run's record beside its streams. A record that cannot be written is said loudly and does
// not change the exit code.
func writeRunRecord(w io.Writer, logDir string, cfg *gateConfig, cells []cellResult, prov provenance, requireFixtures bool) {
	rec := runRecord{Schema: runRecordSchema, Gate: cfg.Name, Decision: cfg.Decision, TopLevelOnly: cfg.TopLevelOnly,
		ZeroPolicy: cfg.ZeroPolicy, RCIsFailure: cfg.RCIsFailure, PkgFailIsFailure: cfg.PkgFailIsFailure,
		RequireFixtures: requireFixtures, Provenance: prov}
	for _, c := range cells {
		rc := recordCell{Name: c.Cell.Name, Pkgs: c.Cell.Pkgs, Tags: c.Cell.Tags, Run: c.Cell.Run, RC: c.RC,
			LogPath: c.LogPath, DurNS: int64(c.Dur)}
		if c.Err != nil {
			rc.Err = c.Err.Error()
		} else if c.LogPath != "" {
			rc.Stream = filepath.Base(c.LogPath)
		}
		rec.Cells = append(rec.Cells, rc)
	}
	b, err := json.MarshalIndent(rec, "", "  ")
	if err == nil {
		err = os.WriteFile(runRecordPath(logDir, cfg), append(b, '\n'), 0o644)
	}
	if err != nil {
		fmt.Fprintf(w, "gate: the run record was not written: %v\n", err)
	}
}

// loadRunRecord reads a record and its streams into what the report functions take: the configuration as the run had
// it (no environment is read), the parsed events, each cell graded by gradeCell, the run's provenance, and whether
// the run required fixtures.
func loadRunRecord(path string) (*gateConfig, *results, []cellResult, provenance, bool, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, nil, provenance{}, false, err
	}
	var rec runRecord
	if err := json.Unmarshal(b, &rec); err != nil {
		return nil, nil, nil, provenance{}, false, fmt.Errorf("%s: %w", path, err)
	}
	if rec.Schema != runRecordSchema {
		return nil, nil, nil, provenance{}, false, fmt.Errorf("%s: schema %d, this gate reads %d", path, rec.Schema, runRecordSchema)
	}
	cfg := &gateConfig{Name: rec.Gate, Decision: rec.Decision, TopLevelOnly: rec.TopLevelOnly, ZeroPolicy: rec.ZeroPolicy,
		RCIsFailure: rec.RCIsFailure, PkgFailIsFailure: rec.PkgFailIsFailure}
	res := newResults()
	var cells []cellResult
	for _, rc := range rec.Cells {
		c := cell{Name: rc.Name, Pkgs: rc.Pkgs, Tags: rc.Tags, Run: rc.Run}
		cfg.Cells = append(cfg.Cells, c)
		if rc.Err != "" {
			cells = append(cells, cellResult{Cell: c, RC: rc.RC, Err: fmt.Errorf("%s", rc.Err), Fail: 1, Hidden: true})
			continue
		}
		f, err := os.Open(filepath.Join(filepath.Dir(path), rc.Stream))
		if err != nil {
			return nil, nil, nil, provenance{}, false, err
		}
		res.cur = rc.Name
		res.consume(f)
		f.Close()
		cells = append(cells, gradeCell(c, cfg, res, rc.RC, rc.LogPath, time.Duration(rc.DurNS)))
	}
	return cfg, res, cells, rec.Provenance, rec.RequireFixtures, nil
}

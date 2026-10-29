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
	Schema           int           `json:"schema"`
	Gate             string        `json:"gate"`
	Decision         string        `json:"decision"`
	TopLevelOnly     bool          `json:"top_level_only"`
	ZeroPolicy       string        `json:"zero_policy"`
	RCIsFailure      bool          `json:"rc_is_failure"`
	PkgFailIsFailure bool          `json:"pkg_fail_is_failure"`
	RequireFixtures  bool          `json:"require_fixtures,omitempty"`
	Provenance       provenance    `json:"provenance"`
	Cells            []recordCell  `json:"cells"`
	Prelude          string        `json:"prelude,omitempty"`  // what the runner printed before the verdict
	Postlude         string        `json:"postlude,omitempty"` // what it printed after (parity: where its summary record went)
	Parity           *parityInputs `json:"parity,omitempty"`   // the parity sweep's inputs beside its streams
	GPU              *gpuRunInputs `json:"gpu,omitempty"`      // the gpu gate's side inputs for the groups that grade a record
}

// runRecordExtra is the part of a record only some gates have.
type runRecordExtra struct {
	Prelude, Postlude string
	Parity            *parityInputs
	GPU               *gpuRunInputs
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
func writeRunRecord(w io.Writer, logDir string, cfg *gateConfig, cells []cellResult, prov provenance, requireFixtures bool, extra runRecordExtra) {
	rec := runRecord{Schema: runRecordSchema, Gate: cfg.Name, Decision: cfg.Decision, TopLevelOnly: cfg.TopLevelOnly,
		ZeroPolicy: cfg.ZeroPolicy, RCIsFailure: cfg.RCIsFailure, PkgFailIsFailure: cfg.PkgFailIsFailure,
		RequireFixtures: requireFixtures, Provenance: prov, Prelude: extra.Prelude, Postlude: extra.Postlude, Parity: extra.Parity, GPU: extra.GPU}
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
	cfg, res, cells, rec, err := loadRunRecordFull(path)
	if err != nil {
		return nil, nil, nil, provenance{}, false, err
	}
	return cfg, res, cells, rec.Provenance, rec.RequireFixtures, nil
}

// loadRunRecordFull is loadRunRecord with the whole record, for the gates whose verdict reads more than its cells.
func loadRunRecordFull(path string) (*gateConfig, *results, []cellResult, *runRecord, error) {
	r, err := readRunRecord(path)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	rec := *r
	cfg := &gateConfig{Name: rec.Gate, Decision: rec.Decision, TopLevelOnly: rec.TopLevelOnly, ZeroPolicy: rec.ZeroPolicy,
		RCIsFailure: rec.RCIsFailure, PkgFailIsFailure: rec.PkgFailIsFailure}
	res := newResults()
	var cells []cellResult
	for _, rc := range rec.Cells {
		c := cell{Name: rc.Name, Pkgs: rc.Pkgs, Tags: rc.Tags, Run: rc.Run}
		cfg.Cells = append(cfg.Cells, c)
		if rc.Err != "" {
			cells = append(cells, gradeCellError(c, rc.RC, fmt.Errorf("%s", rc.Err), rc.LogPath, time.Duration(rc.DurNS)))
			continue
		}
		if rc.Stream == "" {
			return nil, nil, nil, nil, fmt.Errorf("%s: cell %q has neither a stream nor an error", path, rc.Name)
		}
		f, err := os.Open(filepath.Join(filepath.Dir(path), rc.Stream))
		if err != nil {
			return nil, nil, nil, nil, err
		}
		res.cur = rc.Name
		err = res.consume(f)
		f.Close()
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("%s: cell %q's stream %s: %w", path, rc.Name, rc.Stream, err)
		}
		cells = append(cells, gradeCell(c, cfg, res, rc.RC, rc.LogPath, time.Duration(rc.DurNS)))
	}
	return cfg, res, cells, &rec, nil
}

// readRunRecord reads a run record and checks its schema.
func readRunRecord(path string) (*runRecord, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rec runRecord
	if err := json.Unmarshal(b, &rec); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if rec.Schema != runRecordSchema {
		return nil, fmt.Errorf("%s: schema %d, this gate reads %d", path, rec.Schema, runRecordSchema)
	}
	return &rec, nil
}

// replayParity prints a recorded parity sweep's report as the sweep printed it: the runner's prelude, each cell's lines
// from its graded stream, gradeParity on the recorded inputs, and the runner's postlude. It writes no parity record;
// the sweep that ran did, and the postlude says where.
func replayParity(w io.Writer, path string) int {
	cfg, res, cells, rec, err := loadRunRecordFull(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gate: %v\n", err)
		return 2
	}
	if rec.Gate != "parity" || rec.Parity == nil {
		fmt.Fprintf(os.Stderr, "gate: %s is not a parity sweep's record\n", path)
		return 2
	}
	fmt.Fprint(w, rec.Prelude)
	for _, c := range cells {
		fmt.Fprintf(w, "-- running %s ...\n", c.Cell.Name)
		writeParityCell(w, c)
	}
	blockers, _, _ := gradeParity(w, res, cfg.Cells, cells, *rec.Parity)
	fmt.Fprint(w, rec.Postlude)
	if blockers > 0 {
		return 1
	}
	return 0
}

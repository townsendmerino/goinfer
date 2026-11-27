package gatecore

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// RunRecord serializes a gate run's configuration, provenance, cell outcomes, and gate-specific JSON payloads.
type RunRecord struct {
	Schema           int             `json:"schema"`
	Gate             string          `json:"gate"`
	Decision         string          `json:"decision"`
	TopLevelOnly     bool            `json:"top_level_only"`
	ZeroPolicy       string          `json:"zero_policy"`
	RCIsFailure      bool            `json:"rc_is_failure"`
	PkgFailIsFailure bool            `json:"pkg_fail_is_failure"`
	RequireFixtures  bool            `json:"require_fixtures,omitempty"`
	Provenance       Provenance      `json:"provenance"`
	Cells            []RecordCell    `json:"cells"`
	Prelude          string          `json:"prelude,omitempty"`
	Postlude         string          `json:"postlude,omitempty"`
	Parity           json.RawMessage `json:"parity,omitempty"`
	GPU              json.RawMessage `json:"gpu,omitempty"`
	Release          json.RawMessage `json:"release,omitempty"`
	CommentsOnly     json.RawMessage `json:"comments_only,omitempty"`
}

// RunRecordExtra carries optional prelude/postlude text and gate-specific JSON payloads for WriteRunRecord.
type RunRecordExtra struct {
	Prelude, Postlude string
	Parity            any
	GPU               any
	Release           any
	CommentsOnly      any
}

// RecordCell records one executed cell inside a RunRecord.
type RecordCell struct {
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

// RunRecordSchema is the supported schema version for gate_<name>_run.json files.
const RunRecordSchema = 1

// RunRecordPath returns the path where cfg's run record is written inside logDir.
func RunRecordPath(logDir string, cfg *GateConfig) string {
	return filepath.Join(logDir, "gate_"+Sanitize(cfg.Name)+"_run.json")
}

func marshalOptional(v any) (json.RawMessage, error) {
	if v == nil {
		return nil, nil
	}
	b, err := json.Marshal(v)
	if err != nil || string(b) == "null" {
		return nil, err
	}
	return b, nil
}

// UnmarshalPayload decodes a non-empty raw JSON field into a newly allocated T, returning nil when raw is empty or null.
func UnmarshalPayload[T any](raw json.RawMessage) (*T, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var out T
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// WriteRunRecord writes cfg's RunRecord beside its cell streams in logDir, reporting write errors to w without failing the gate.
func WriteRunRecord(w io.Writer, logDir string, cfg *GateConfig, cells []CellResult, prov Provenance, requireFixtures bool, extra RunRecordExtra) {
	parityRaw, err := marshalOptional(extra.Parity)
	var gpuRaw, relRaw, coRaw json.RawMessage
	if err == nil {
		gpuRaw, err = marshalOptional(extra.GPU)
	}
	if err == nil {
		relRaw, err = marshalOptional(extra.Release)
	}
	if err == nil {
		coRaw, err = marshalOptional(extra.CommentsOnly)
	}
	if err != nil {
		fmt.Fprintf(w, "gate: the run record was not written: %v\n", err)
		return
	}
	rec := RunRecord{
		Schema: RunRecordSchema, Gate: cfg.Name, Decision: cfg.Decision, TopLevelOnly: cfg.TopLevelOnly,
		ZeroPolicy: cfg.ZeroPolicy, RCIsFailure: cfg.RCIsFailure, PkgFailIsFailure: cfg.PkgFailIsFailure,
		RequireFixtures: requireFixtures, Provenance: prov, Prelude: extra.Prelude, Postlude: extra.Postlude,
		Parity: parityRaw, GPU: gpuRaw, Release: relRaw, CommentsOnly: coRaw,
	}
	for _, c := range cells {
		rc := RecordCell{
			Name: c.Cell.Name, Pkgs: c.Cell.Pkgs, Tags: c.Cell.Tags, Run: c.Cell.Run, RC: c.RC,
			LogPath: c.LogPath, DurNS: int64(c.Dur),
		}
		if c.Err != nil {
			rc.Err = c.Err.Error()
		} else if c.LogPath != "" {
			rc.Stream = filepath.Base(c.LogPath)
		}
		rec.Cells = append(rec.Cells, rc)
	}
	b, err := json.MarshalIndent(rec, "", "  ")
	if err == nil {
		err = os.WriteFile(RunRecordPath(logDir, cfg), append(b, '\n'), 0o644)
	}
	if err != nil {
		fmt.Fprintf(w, "gate: the run record was not written: %v\n", err)
	}
}

// LoadRunRecord reads a recorded run and its cell streams for replay grading.
func LoadRunRecord(path string) (*GateConfig, *Results, []CellResult, Provenance, bool, error) {
	cfg, res, cells, rec, err := LoadRunRecordFull(path)
	if err != nil {
		return nil, nil, nil, Provenance{}, false, err
	}
	return cfg, res, cells, rec.Provenance, rec.RequireFixtures, nil
}

// LoadRunRecordFull reads a recorded run, its streams, and the full RunRecord envelope.
func LoadRunRecordFull(path string) (*GateConfig, *Results, []CellResult, *RunRecord, error) {
	r, err := ReadRunRecord(path)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	rec := *r
	cfg := &GateConfig{
		Name: rec.Gate, Decision: rec.Decision, TopLevelOnly: rec.TopLevelOnly, ZeroPolicy: rec.ZeroPolicy,
		RCIsFailure: rec.RCIsFailure, PkgFailIsFailure: rec.PkgFailIsFailure,
	}
	res := NewResults()
	var cells []CellResult
	for _, rc := range rec.Cells {
		c := Cell{Name: rc.Name, Pkgs: rc.Pkgs, Tags: rc.Tags, Run: rc.Run}
		cfg.Cells = append(cfg.Cells, c)
		if rc.Err != "" {
			cells = append(cells, GradeCellError(c, rc.RC, fmt.Errorf("%s", rc.Err), rc.LogPath, time.Duration(rc.DurNS)))
			continue
		}
		if rc.Stream == "" {
			return nil, nil, nil, nil, fmt.Errorf("%s: cell %q has neither a stream nor an error", path, rc.Name)
		}
		f, err := os.Open(filepath.Join(filepath.Dir(path), rc.Stream))
		if err != nil {
			return nil, nil, nil, nil, err
		}
		res.Cur = rc.Name
		err = res.Consume(f)
		f.Close()
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("%s: cell %q's stream %s: %w", path, rc.Name, rc.Stream, err)
		}
		cells = append(cells, GradeCell(c, cfg, res, rc.RC, rc.LogPath, time.Duration(rc.DurNS)))
	}
	return cfg, res, cells, &rec, nil
}

// ReadRunRecord reads and validates a RunRecord JSON (or .gz) file without opening cell streams.
func ReadRunRecord(path string) (*RunRecord, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if strings.HasSuffix(path, ".gz") {
		zr, err := gzip.NewReader(bytes.NewReader(b))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		if b, err = io.ReadAll(zr); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
	}
	var rec RunRecord
	if err := json.Unmarshal(b, &rec); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if rec.Schema != RunRecordSchema {
		return nil, fmt.Errorf("%s: schema %d, this gate reads %d", path, rec.Schema, RunRecordSchema)
	}
	return &rec, nil
}

// WriteFileAtomic writes b to path via a sibling .tmp file and rename.
func WriteFileAtomic(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

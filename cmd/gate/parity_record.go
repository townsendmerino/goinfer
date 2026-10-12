package main

import "io"

// The parity sweep's record (the §C1 row of `gate release`). This file is the stub the row's tests were registered
// against (parity_record_test.go); nothing here decides anything yet.

const parityRecordDirDefault = "docs/measurements/parity-sweep"

const parityManifestPath = "testdata/parity_manifest.json"

type parityRecordCell struct {
	Name             string `json:"name"`
	Pass, Skip, Fail int
	RC               int
}

type parityRecord struct {
	Schema    int                `json:"schema"`
	Commit    string             `json:"commit"`
	Date      string             `json:"date"`
	Host      string             `json:"host"`
	Verdict   string             `json:"verdict"`
	Dirty     bool               `json:"dirty"`
	Realckpt  bool               `json:"realckpt"`
	Scoped    string             `json:"scoped,omitempty"`
	Required  int                `json:"required"`
	Blockers  int                `json:"blockers"`
	Gaps      int                `json:"gaps"`
	FirstRuns int                `json:"first_runs"`
	Cells     []parityRecordCell `json:"cells"`
	Paths     map[string]string  `json:"paths"`
}

type parityRun struct {
	Realckpt                            bool
	Scoped                              string
	Required, Blockers, Gaps, FirstRuns int
	Cells                               []cellResult
}

func parityQualifyingPaths(manifest []byte) ([]string, error) { return nil, nil }

func gitObject(root, rev, path string) (string, error) { return "", nil }

func parityCoverage(recs []parityRecord, head string, paths []string, objOf func(rev, path string) (string, error)) (*parityRecord, string) {
	return nil, "not implemented"
}

func loadParityRecords(dir string) ([]parityRecord, error) { return nil, nil }

func writeParityRecord(w io.Writer, root, recordDir string, run parityRun) {}

func rowParity(e *relEnv) relRow {
	return owed("the §C1 parity sweep covers HEAD", "", "not implemented")
}

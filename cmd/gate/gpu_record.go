package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// The GPU gate's record. `gate gpu` prints a verdict and keeps its raw stream in the log directory, and nothing durable said
// "cuda, this commit, PASS": RELEASING.md's read 8 was prose, and v0.21.0 and v0.22.0 shipped with the Metal run and no recorded
// CUDA one. Every run now writes one small JSON file, committed beside the other measurement records, and `gate gpu-records`
// reads them: a backend is covered at a commit when its newest passing record's commit has the same tree under each of
// gpuQualifyingTrees as that commit does, which is RELEASING.md's rule ("a run at an earlier commit counts only if nothing under
// metal/, cuda/, gpu/ or decoder/ changed since") made checkable.

// gpuRecordDirDefault is where a run's record goes, relative to the repository root.
const gpuRecordDirDefault = "docs/measurements/gpu-gate"

// gpuRecord is one run of `gate gpu`, whatever its verdict. Only a PASS on a clean tree with the heavy tier run can qualify.
type gpuRecord struct {
	Schema       int               `json:"schema"`
	Backend      string            `json:"backend"`
	Commit       string            `json:"commit"` // the full SHA the run was made at
	Date         string            `json:"date"`   // UTC, RFC 3339
	Host         string            `json:"host"`
	GPU          string            `json:"gpu,omitempty"`
	Verdict      string            `json:"verdict"` // PASS, FAIL, INCONCLUSIVE (green on a dirty tree), NO_GATE, VACUOUS, EMPTY
	Dirty        bool              `json:"dirty"`
	HeavySkipped bool              `json:"heavy_skipped"`
	Groups       gpuRecordGroups   `json:"groups"`
	Trees        map[string]string `json:"trees"` // git tree id of each gpuQualifyingTrees directory at Commit, for a reader; qualification recomputes them
	// HeavyProcesses is each heavy-tier main-group process's memory, as the never-swap rule reads it.
	HeavyProcesses []heavyProcRecord `json:"heavy_processes,omitempty"`
	// The tally gradeGPU decided from, by name, so a record regrades in full; records before SB12 step 7 hold only
	// Groups' counts.
	Expect       []string `json:"expect,omitempty"`
	Emitted      []string `json:"emitted,omitempty"`
	VacuousCells []string `json:"vacuous_cells,omitempty"`
	EmptyCells   []string `json:"empty_cells,omitempty"`
	Notes        []string `json:"notes,omitempty"`
}

// heavyProcRecord is one heavy-tier process: its group, how many tests it owned, its wall time, and its test binary's
// VmHWM and peak VmSwap in MB (procPeak). Samples is 0 where /proc could not be read, and the two figures are then absent,
// not zero.
type heavyProcRecord struct {
	Group      string `json:"group"`
	Tests      int    `json:"tests"`
	WallS      int    `json:"wall_s"`
	Samples    int    `json:"samples"`
	VmHWMMB    int64  `json:"vmhwm_mb,omitempty"`
	VmSwapPkMB int64  `json:"vmswap_peak_mb,omitempty"`
}

type gpuRecordGroups struct {
	Declared int `json:"declared"`
	Reported int `json:"reported"`
	Pass     int `json:"pass"`
	Skip     int `json:"skip"`
	Fail     int `json:"fail"`
	Ran      int `json:"ran"`
}

// writeRecord writes this run's record. A record that cannot be written is said loudly and does not change the exit code: the
// run's verdict is what it is, and the release command will read the missing record as owed.
func (g *gpuGate) writeRecord(verdict string) {
	if g.recordDir == "" {
		return
	}
	root, err := repoRoot()
	if err != nil {
		fmt.Fprintf(g.w, "\n  RECORD NOT WRITTEN: %v\n", err)
		return
	}
	dir := g.recordDir
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(root, dir)
	}
	full, err := gitOut(root, "rev-parse", "HEAD")
	if err != nil {
		fmt.Fprintf(g.w, "\n  RECORD NOT WRITTEN: %v\n", err)
		return
	}
	now := time.Now().UTC()
	rec := gpuRecord{
		Schema: 1, Backend: g.backend, Commit: full, Date: now.Format(time.RFC3339), Verdict: verdict, Dirty: g.dirty, HeavySkipped: g.heavySkipped,
		GPU: g.gpuName, Trees: map[string]string{}, HeavyProcesses: g.heavyProcs,
		Groups: gpuRecordGroups{Declared: len(g.expect), Reported: len(g.emitted), Pass: g.pass, Skip: g.skipped, Fail: g.fail, Ran: g.ran},
		Expect: g.expect, Emitted: sortedSet(g.emitted), VacuousCells: g.vacuousCells, EmptyCells: g.emptyCells, Notes: g.notes,
	}
	if b, err := exec.Command("uname", "-sm").Output(); err == nil {
		rec.Host = strings.TrimSpace(string(b))
	}
	for _, d := range gpuQualifyingTrees {
		if t, err := gitTree(root, full, d); err == nil {
			rec.Trees[d] = t
		}
	}
	b, err := json.MarshalIndent(rec, "", " ")
	if err != nil {
		fmt.Fprintf(g.w, "\n  RECORD NOT WRITTEN: %v\n", err)
		return
	}
	name := fmt.Sprintf("%s-%s-%s.json", g.backend, now.Format("20060102T150405Z"), full[:8])
	path := filepath.Join(dir, name)
	if err := writeFileAtomic(path, append(b, '\n')); err != nil {
		fmt.Fprintf(g.w, "\n  RECORD NOT WRITTEN: %v\n", err)
		return
	}
	fmt.Fprintf(g.w, "\n  record: %s (commit it with the release; `gate gpu-records` reads it)\n", path)
}

// writeFileAtomic writes b to path through a temporary file and a rename, so a reader never sees half a record.
func writeFileAtomic(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// gitTree is the git tree id of dir at rev.
func gitTree(root, rev, dir string) (string, error) {
	return gitOut(root, "rev-parse", rev+":"+dir)
}

// loadGPURecords reads every record in dir, oldest first. A file that is not a record is an error, not a skip: a record that
// cannot be read must not read as "no record".
func loadGPURecords(dir string) ([]gpuRecord, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var recs []gpuRecord
	for _, e := range ents {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, err
		}
		var r gpuRecord
		if err := json.Unmarshal(b, &r); err != nil || r.Schema != 1 || r.Backend == "" || r.Commit == "" {
			return nil, fmt.Errorf("%s is not a GPU-gate record (schema 1 with a backend and a commit): %v", e.Name(), err)
		}
		recs = append(recs, r)
	}
	sort.SliceStable(recs, func(i, j int) bool { return recs[i].Date < recs[j].Date })
	return recs, nil
}

// runGPURecords is `gate gpu-records`: for each backend, whether a GPU-gate run still covers head. Exit 0 only when both do.
func runGPURecords(args []string, w io.Writer) int {
	fs := flag.NewFlagSet("gate gpu-records", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	head := fs.String("head", "HEAD", "the revision to be covered")
	dir := fs.String("dir", gpuRecordDirDefault, "where the records are")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	root, err := repoRoot()
	if err != nil {
		fmt.Fprintln(w, err)
		return 2
	}
	d := *dir
	if !filepath.IsAbs(d) {
		d = filepath.Join(root, d)
	}
	recs, err := loadGPURecords(d)
	if err != nil {
		fmt.Fprintf(w, "gate gpu-records: %v\n", err)
		return 2
	}
	full, err := gitOut(root, "rev-parse", *head)
	if err != nil {
		fmt.Fprintf(w, "gate gpu-records: %v\n", err)
		return 2
	}
	treeOf := func(rev, dir string) (string, error) { return gitTree(root, rev, dir) }
	code := 0
	fmt.Fprintf(w, "GPU gate records in %s, against %s\n", *dir, full[:8])
	for _, be := range []string{"cuda", "metal"} {
		r, why := gpuCoverage(recs, be, full, treeOf)
		if r != nil {
			fmt.Fprintf(w, "  %-6s done   %s @ %s (%d pass, %d skip)\n", be, r.Date, r.Commit[:8], r.Groups.Pass, r.Groups.Skip)
			continue
		}
		code = 1
		fmt.Fprintf(w, "  %-6s OWED   %s\n", be, why)
	}
	return code
}

package main

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// gpuRunInputs is what the gpu gate's moved groups read beside their cells: the backend, and the Metal checkpoint the
// lifecycle and prefill groups need, with whether it was present when the run looked.
type gpuRunInputs struct {
	Backend           string `json:"backend"`
	MetalModel        string `json:"metal_model,omitempty"`
	MetalModelPresent bool   `json:"metal_model_present,omitempty"`
	// CgoFree and WebGPU are the cgofree group's build and linkage and the WebGPU adapter probe; a record before SB12 step 9
	// holds neither, and the replay then does not regrade those groups.
	CgoFree *cgoFreeInput      `json:"cgo_free,omitempty"`
	WebGPU  *webgpuProbeRecord `json:"webgpu,omitempty"`
}

// gpuRunConfig is the configuration every gpu-gate cell runs and is graded under (gpuGate.run).
func gpuRunConfig() *gateConfig {
	return &gateConfig{Name: "gpu", TopLevelOnly: true, RCIsFailure: true}
}

// writeGPURunRecord writes the run record (gate_gpu_run.json) beside the cells' streams, when the run has a log directory.
func (g *gpuGate) writeGPURunRecord() {
	if g.logDir == "" {
		return
	}
	in := g.runIn
	in.Backend = g.backend
	writeRunRecord(g.w, g.logDir, gpuRunConfig(), g.cells, g.prov, false, runRecordExtra{GPU: &in})
}

// gpuMetalGroups is every group a Metal run declares, in order.
var gpuMetalGroups = []string{"cleangpu", "seam", "suite", "parity", "cgofree", "lifecycle", "prefill", "webgpu", "repo"}

// replayGPU regrades the gpu gate's moved groups from a run record through the same sequence the run used (metalGroups,
// webgpuGroup) and prints their sections as the run did, then names the groups it did not regrade. It gives no gate
// verdict, which needs every group: it exits 1 when a regraded group failed or a regraded cell was empty or all-skip (the
// live gate's tally fails those even where the group passed), 0 otherwise, and 2 when the record cannot be read.
func replayGPU(w io.Writer, path string) int {
	want := map[string]bool{webgpuSuiteCell.Name: true, webgpuParityCell.Name: true}
	for _, s := range metalCellGroups("") {
		want[s.Cell.Name] = true
	}
	rec, cells, texts, err := loadGPURunRecord(path, want)
	if err != nil {
		fmt.Fprintf(w, "gate gpu -replay: %v\n", err)
		return 2
	}
	if rec.GPU == nil || rec.GPU.Backend != "metal" {
		fmt.Fprintf(w, "gate gpu -replay: %s is not a Metal run record; only the Metal groups regrade from a record yet\n", path)
		return 2
	}
	g := &gpuGate{w: w, backend: rec.GPU.Backend, emitted: map[string]bool{}, runIn: *rec.GPU}
	src := recordedInputs{g: g, path: path, cells: cells, texts: texts}
	err = g.metalGroups(src)
	if err == nil {
		err = g.webgpuGroup(src)
	}
	if err != nil {
		fmt.Fprintf(w, "gate gpu -replay: %v\n", err)
		return 2
	}
	var done, rest []string
	for _, n := range gpuMetalGroups {
		if g.emitted[n] {
			done = append(done, n)
		} else {
			rest = append(rest, n)
		}
	}
	fmt.Fprintf(w, "\nregraded %d group(s) (%d fail, %d all-skip cell(s), %d empty cell(s)); not regraded from a record: %s. "+
		"This is not the gate's verdict.\n",
		len(done), g.fail, len(g.vacuousCells), len(g.emptyCells), strings.Join(rest, ", "))
	if g.fail > 0 || len(g.vacuousCells) > 0 || len(g.emptyCells) > 0 {
		return 1
	}
	return 0
}

// recordedInputs reads the groups' cells and side inputs from a run record. A cell is graded as the run graded it, with the
// runner's own empty and all-skip notes.
type recordedInputs struct {
	g     *gpuGate
	path  string
	cells map[string]gpuRecordedCell
	texts map[string]string
}

func (r recordedInputs) cell(c cell) (cellResult, string, error) {
	i, ok := r.cells[c.Name]
	if !ok {
		return cellResult{}, "", fmt.Errorf("%s holds no %s cell, or its stream is not beside it", r.path, c.Name)
	}
	cr := gradeCell(c, gpuRunConfig(), i.res, i.rc.RC, i.rc.LogPath, time.Duration(i.rc.DurNS))
	if i.rc.Err != "" {
		cr = gradeCellError(c, i.rc.RC, fmt.Errorf("%s", i.rc.Err), i.rc.LogPath, time.Duration(i.rc.DurNS))
	}
	r.g.noteIfEmpty(c, i.res)
	r.g.noteIfAllSkipped(c, cr)
	return cr, r.texts[c.Name], nil
}

func (r recordedInputs) cgoFree() (*cgoFreeInput, error) { return r.g.runIn.CgoFree, nil }

func (r recordedInputs) webgpuProbe() *webgpuProbeRecord { return r.g.runIn.WebGPU }

// gpuRecordedCell is one recorded cell with its own parsed stream, so each group reads only its cell's text.
type gpuRecordedCell struct {
	rc  recordCell
	res *results
}

// loadGPURunRecord reads a gpu run record and parses the stream of each cell named in want into its own results. A cell
// whose stream file is absent is left out.
func loadGPURunRecord(path string, want map[string]bool) (*runRecord, map[string]gpuRecordedCell, map[string]string, error) {
	rec, err := readRunRecord(path)
	if err != nil {
		return nil, nil, nil, err
	}
	cells, texts := map[string]gpuRecordedCell{}, map[string]string{}
	for _, rc := range rec.Cells {
		if !want[rc.Name] {
			continue
		}
		res := newResults()
		res.cur = rc.Name
		if rc.Err == "" {
			if rc.Stream == "" {
				return nil, nil, nil, fmt.Errorf("%s: cell %q has neither a stream nor an error", path, rc.Name)
			}
			f, err := os.Open(filepath.Join(filepath.Dir(path), rc.Stream))
			if errors.Is(err, fs.ErrNotExist) {
				continue // not kept beside the record: an error only if a group asks for this cell
			}
			if err != nil {
				return nil, nil, nil, err
			}
			err = res.consume(f)
			f.Close()
			if err != nil {
				return nil, nil, nil, fmt.Errorf("%s: cell %q's stream %s: %w", path, rc.Name, rc.Stream, err)
			}
		}
		cells[rc.Name] = gpuRecordedCell{rc: rc, res: res}
		texts[rc.Name] = res.text()
	}
	return rec, cells, texts, nil
}

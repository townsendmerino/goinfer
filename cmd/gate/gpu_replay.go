package main

import (
	"fmt"
	"io"
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

// gpuReplayGroups is the groups `gate gpu -replay` regrades, per backend; the rest still grade inside the runner.
var gpuReplayGroups = map[string][]string{"metal": {"suite", "parity", "lifecycle", "prefill"}}

// replayGPU regrades the gpu gate's moved groups from a run record and prints their sections as the run did, then names
// the groups it did not regrade. It gives no gate verdict, which needs every group: it exits 1 when a regraded group
// failed, 0 otherwise, and 2 when the record cannot be read.
func replayGPU(w io.Writer, path string) int {
	want := map[string]bool{}
	for _, s := range metalCellGroups("") {
		want[s.Cell.Name] = true
	}
	rec, cells, texts, err := loadGPURunRecord(path, want)
	if err != nil {
		fmt.Fprintf(w, "gate gpu -replay: %v\n", err)
		return 2
	}
	if rec.GPU == nil || rec.GPU.Backend != "metal" {
		fmt.Fprintf(w, "gate gpu -replay: %s is not a Metal run record; only the Metal cell groups regrade from a record yet\n", path)
		return 2
	}
	g := &gpuGate{w: w, backend: rec.GPU.Backend, emitted: map[string]bool{}, runIn: *rec.GPU}
	for _, s := range metalCellGroups(rec.GPU.MetalModel) {
		g.grp(s.Group)
		g.hdr(s.Header)
		if s.NeedsModel && !rec.GPU.MetalModelPresent {
			g.apply(gradeMetalCell(s, rec.GPU.MetalModel, false, cellResult{}, ""))
			continue
		}
		i, ok := cells[s.Cell.Name]
		if !ok {
			fmt.Fprintf(w, "gate gpu -replay: %s holds no %s cell\n", path, s.Cell.Name)
			return 2
		}
		cr := gradeCell(s.Cell, gpuRunConfig(), i.res, i.rc.RC, i.rc.LogPath, time.Duration(i.rc.DurNS))
		if i.rc.Err != "" {
			cr = gradeCellError(s.Cell, i.rc.RC, fmt.Errorf("%s", i.rc.Err), i.rc.LogPath, time.Duration(i.rc.DurNS))
		}
		g.noteIfEmpty(s.Cell, i.res)
		g.noteIfAllSkipped(s.Cell, cr)
		g.apply(gradeMetalCell(s, rec.GPU.MetalModel, true, cr, texts[s.Cell.Name]))
	}
	var rest []string
	moved := map[string]bool{}
	for _, n := range gpuReplayGroups["metal"] {
		moved[n] = true
	}
	for _, n := range []string{"cleangpu", "seam", "suite", "parity", "cgofree", "lifecycle", "prefill", "webgpu", "repo"} {
		if !moved[n] {
			rest = append(rest, n)
		}
	}
	fmt.Fprintf(w, "\nregraded %d group(s) (%d fail); not regraded from a record yet: %s. This is not the gate's verdict.\n",
		len(gpuReplayGroups["metal"]), g.fail, strings.Join(rest, ", "))
	if g.fail > 0 {
		return 1
	}
	return 0
}

// gpuRecordedCell is one recorded cell with its own parsed stream, so each group reads only its cell's text.
type gpuRecordedCell struct {
	rc  recordCell
	res *results
}

// loadGPURunRecord reads a gpu run record and parses the stream of each cell named in want into its own results.
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

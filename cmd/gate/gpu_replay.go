package main

import (
	"compress/gzip"
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
	// ParityFreeVRAM is the card's free memory the CUDA runner read after a failed parity cell whose text held a cosine of
	// exactly zero (gradeCUDAParity), as nvidia-smi printed it; "" when it did not read it.
	ParityFreeVRAM string `json:"parity_free_vram,omitempty"`
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

// gpuMetalGroups and gpuCUDAGroups are every group a Metal or CUDA run declares, in order.
var (
	gpuMetalGroups = []string{"cleangpu", "seam", "suite", "parity", "cgofree", "lifecycle", "prefill", "webgpu", "repo"}
	gpuCUDAGroups  = []string{"cleangpu", "seam", "suite", "parity", "heavy", "graphsforced", "cgofree", "ptx", "webgpu", "repo"}
)

// replayGPU regrades the gpu gate's moved groups from a run record through the same sequence the run used (metalGroups or
// cudaGroups, then webgpuGroup) and prints their sections as the run did, then names the groups it did not regrade. It gives no gate
// verdict, which needs every group: it exits 1 when a regraded group failed or a regraded cell was empty or all-skip (the
// live gate's tally fails those even where the group passed), 0 otherwise, and 2 when the record cannot be read.
func replayGPU(w io.Writer, path string) int {
	want := map[string]bool{webgpuSuiteCell.Name: true, webgpuParityCell.Name: true,
		cudaSuiteCell.Name: true, cudaParityCell.Name: true, cudaGraphsCell.Name: true}
	for _, s := range metalCellGroups("") {
		want[s.Cell.Name] = true
	}
	rec, cells, texts, err := loadGPURunRecord(path, want)
	if err != nil {
		fmt.Fprintf(w, "gate gpu -replay: %v\n", err)
		return 2
	}
	if rec.GPU == nil || (rec.GPU.Backend != "metal" && rec.GPU.Backend != "cuda") {
		fmt.Fprintf(w, "gate gpu -replay: %s is not a Metal or CUDA run record; only those groups regrade from a record\n", path)
		return 2
	}
	g := &gpuGate{w: w, backend: rec.GPU.Backend, emitted: map[string]bool{}, runIn: *rec.GPU}
	src := recordedInputs{g: g, path: path, cells: cells, texts: texts}
	groups := gpuMetalGroups
	if g.backend == "cuda" {
		groups = gpuCUDAGroups
		err = g.cudaGroups(src, false)
	} else {
		err = g.metalGroups(src)
	}
	if err == nil {
		err = g.webgpuGroup(src)
	}
	if err != nil {
		fmt.Fprintf(w, "gate gpu -replay: %v\n", err)
		return 2
	}
	var done, rest []string
	for _, n := range groups {
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

func (r recordedInputs) cell(c cell) (*results, cellResult, string, error) {
	i, ok := r.cells[c.Name]
	if !ok {
		return nil, cellResult{}, "", fmt.Errorf("%s holds no %s cell, or its stream is not beside it", r.path, c.Name)
	}
	cr := gradeCell(c, gpuRunConfig(), i.res, i.rc.RC, i.rc.LogPath, time.Duration(i.rc.DurNS))
	if i.rc.Err != "" {
		cr = gradeCellError(c, i.rc.RC, fmt.Errorf("%s", i.rc.Err), i.rc.LogPath, time.Duration(i.rc.DurNS))
	}
	r.g.noteIfEmpty(c, i.res)
	r.g.noteIfAllSkipped(c, cr)
	return i.res, cr, r.texts[c.Name], nil
}

func (r recordedInputs) cgoFree() (*cgoFreeInput, error) { return r.g.runIn.CgoFree, nil }

func (r recordedInputs) webgpuProbe() *webgpuProbeRecord { return r.g.runIn.WebGPU }

func (r recordedInputs) parityFreeVRAM(cellResult, string) string { return r.g.runIn.ParityFreeVRAM }

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
			r, err := openStream(filepath.Join(filepath.Dir(path), rc.Stream))
			if errors.Is(err, fs.ErrNotExist) {
				continue // not kept beside the record: an error only if a group asks for this cell
			}
			if err != nil {
				return nil, nil, nil, err
			}
			err = res.consume(r)
			r.Close()
			if err != nil {
				return nil, nil, nil, fmt.Errorf("%s: cell %q's stream %s: %w", path, rc.Name, rc.Stream, err)
			}
		}
		cells[rc.Name] = gpuRecordedCell{rc: rc, res: res}
		texts[rc.Name] = res.text()
	}
	return rec, cells, texts, nil
}

// openStream opens a recorded cell stream, or its gzipped copy (name + ".gz") when the plain file is absent: a stream over
// 1 MB is committed gzipped. Closing the reader closes the file.
func openStream(name string) (io.ReadCloser, error) {
	f, err := os.Open(name)
	if err == nil || !errors.Is(err, fs.ErrNotExist) {
		return f, err
	}
	gf, gerr := os.Open(name + ".gz")
	if gerr != nil {
		return nil, err // the plain name's not-exist error
	}
	zr, err := gzip.NewReader(gf)
	if err != nil {
		gf.Close()
		return nil, fmt.Errorf("%s.gz: %w", name, err)
	}
	return gzipFile{zr, gf}, nil
}

// gzipFile is a gzip reader that closes its file.
type gzipFile struct {
	*gzip.Reader
	f *os.File
}

func (z gzipFile) Close() error {
	z.Reader.Close()
	return z.f.Close()
}

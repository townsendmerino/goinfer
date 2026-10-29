package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const metalPkg = "github.com/townsendmerino/goinfer/metal"

// metalCellEvents is a cell's stream: each test run with action act and one output line, then the package's ok or FAIL
// line. A crash adds a panic and a frame from this repo after the first test.
func metalCellEvents(tests []string, act string, crash bool) []testEvent {
	var evs []testEvent
	for i, name := range tests {
		evs = append(evs, testEvent{Action: "run", Package: metalPkg, Test: name},
			testEvent{Action: "output", Package: metalPkg, Test: name, Output: "    x_test.go:9: trajectory argmax matches " + act + "\n"})
		if crash && i == 0 {
			evs = append(evs, testEvent{Action: "output", Package: metalPkg, Output: "panic: boom\n"},
				testEvent{Action: "output", Package: metalPkg, Output: "goroutine 1 [running]:\n"},
				testEvent{Action: "output", Package: metalPkg, Output: "github.com/townsendmerino/goinfer/metal.TestX(...)\n"})
			return evs
		}
		if act == "fail" {
			evs = append(evs, testEvent{Action: "output", Package: metalPkg, Test: name, Output: "--- FAIL: " + name + " (0.00s)\n"},
				testEvent{Action: "output", Package: metalPkg, Test: name, Output: "    x_test.go:12: got 1, want 2\n"})
		}
		evs = append(evs, testEvent{Action: act, Package: metalPkg, Test: name})
	}
	line := "ok  \t" + metalPkg + "\t1.000s\n"
	if act == "fail" {
		line = "FAIL\t" + metalPkg + "\t1.000s\n"
	}
	return append(evs, testEvent{Action: "output", Package: metalPkg, Output: line})
}

// metalCellTests is two test names that match each group's -run, so a filtered cell is not empty.
func metalCellTests(g metalCellGroup) []string {
	if g.Cell.Run == "" {
		return []string{"TestA", "TestB"}
	}
	alts := strings.Split(g.Cell.Run, "|")
	return []string{alts[0], alts[len(alts)-1]}
}

func gradeEvents(c cell, evs []testEvent, rc int) (*results, cellResult) {
	res := newResults()
	res.cur = c.Name
	for _, ev := range evs {
		res.add(ev)
	}
	return res, gradeCell(c, gpuRunConfig(), res, rc, "", 0)
}

// TestGradeMetalCell_plants is SB12-C4 for step 8 (docs/tasks/task-subsystems-2026-10.md, SB12 step 8 (Metal)): from a
// green cell each group passes with its evidence; a failed test fails each group; an all-skip cell fails every group that
// VacuousFails and the suite still passes; an absent checkpoint skips with no cell; a crash fails with the crash head.
func TestGradeMetalCell_plants(t *testing.T) {
	const model = "/m/qwen.gguf"
	for _, g := range metalCellGroups(model) {
		tests := metalCellTests(g)
		res, cr := gradeEvents(g.Cell, metalCellEvents(tests, "pass", false), 0)
		v := gradeMetalCell(g, model, true, cr, res.text())
		if v.State != groupPass || v.Line != g.OK {
			t.Errorf("%s green: state %d line %q", g.Group, v.State, v.Line)
		}
		if g.Evidence != nil && len(v.Lines) == 0 {
			t.Errorf("%s green: no evidence line", g.Group)
		}

		res, cr = gradeEvents(g.Cell, metalCellEvents(tests, "fail", false), 1)
		v = gradeMetalCell(g, model, true, cr, res.text())
		detail := strings.Join(v.Lines, "\n")
		if v.State != groupFail || v.Line != g.Bad || !strings.Contains(detail, "--- FAIL: "+tests[0]) ||
			!strings.Contains(detail, "x_test.go:12: got 1, want 2") || strings.Contains(detail, "NO ASSERTION LINE MATCHED") {
			t.Errorf("%s with a failed test: state %d line %q, detail:\n%s\nwant FAIL naming the failed test and its assertion", g.Group, v.State, v.Line, detail)
		}

		res, cr = gradeEvents(g.Cell, metalCellEvents(tests, "skip", false), 0)
		v = gradeMetalCell(g, model, true, cr, res.text())
		if want := map[bool]groupState{true: groupFail, false: groupPass}[g.VacuousFails]; v.State != want {
			t.Errorf("%s with every test skipped: state %d, want %d", g.Group, v.State, want)
		}

		res, cr = gradeEvents(g.Cell, metalCellEvents(tests, "pass", true), 2)
		v = gradeMetalCell(g, model, true, cr, res.text())
		if v.State != groupFail || len(v.Lines) == 0 || !strings.Contains(v.Lines[0], "PROCESS DIED") {
			t.Errorf("%s with a crash: state %d, lines %q; want FAIL with the crash head", g.Group, v.State, v.Lines)
		}

		v = gradeMetalCell(g, model, false, cellResult{}, "")
		if g.NeedsModel {
			if v.State != groupSkip || v.Line != strings.Replace(g.Missing, "%s", model, 1) {
				t.Errorf("%s with no checkpoint: state %d line %q, want SKIP naming %s", g.Group, v.State, v.Line, model)
			}
		} else if v.State != groupPass {
			t.Errorf("%s does not need the checkpoint, but its absence gave state %d", g.Group, v.State)
		}
	}
}

// fakeLiveInputs stands in for liveInputs at tier 0: each cell's stream is written where runCell writes it and graded as
// gpuGate.run grades it (with its empty and all-skip notes); the side inputs are given.
type fakeLiveInputs struct {
	t     *testing.T
	g     *gpuGate
	evs   map[string][]testEvent // by cell name
	rcs   map[string]int
	cgo   cgoFreeInput
	asked map[string]bool
}

func (f fakeLiveInputs) cell(c cell) (*results, cellResult, string, error) {
	f.asked[c.Name] = true
	path := filepath.Join(f.g.logDir, "gate_"+sanitize("gpu_"+c.Name)+".json")
	var b bytes.Buffer
	for _, ev := range f.evs[c.Name] {
		j, _ := json.Marshal(ev)
		b.Write(append(j, '\n'))
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		f.t.Fatal(err)
	}
	res, cr := gradeEvents(c, f.evs[c.Name], f.rcs[c.Name])
	cr.LogPath = path
	f.g.cells = append(f.g.cells, cr)
	f.g.noteIfEmpty(c, res)
	f.g.noteIfAllSkipped(c, cr)
	return res, cr, res.text(), nil
}

func (f fakeLiveInputs) cgoFree() (*cgoFreeInput, error) {
	in := f.cgo
	f.g.runIn.CgoFree = &in
	return &in, nil
}

func (f fakeLiveInputs) webgpuProbe() *webgpuProbeRecord { return f.g.runIn.WebGPU }

func (f fakeLiveInputs) parityFreeVRAM(cellResult, string) string { return f.g.runIn.ParityFreeVRAM }

// recordMetalRun runs the Metal groups and group W through the gate's own sequence on a fake live source, writes the run
// record, and returns what the run printed, the record's path and the cells the sequence asked for.
func recordMetalRun(t *testing.T, evs map[string][]testEvent, rcs map[string]int, cgo cgoFreeInput, probe webgpuProbeRecord) (string, string, map[string]bool) {
	t.Helper()
	dir := t.TempDir()
	var live bytes.Buffer
	model := filepath.Join(dir, "qwen.gguf")
	g := &gpuGate{w: &live, logDir: dir, backend: "metal", emitted: map[string]bool{},
		runIn: gpuRunInputs{MetalModel: model, MetalModelPresent: true, WebGPU: &probe}}
	src := fakeLiveInputs{t: t, g: g, evs: evs, rcs: rcs, cgo: cgo, asked: map[string]bool{}}
	if err := g.metalGroups(src); err != nil {
		t.Fatal(err)
	}
	if err := g.webgpuGroup(src); err != nil {
		t.Fatal(err)
	}
	runOut := live.String()
	g.writeGPURunRecord()
	if live.Len() != len(runOut) {
		t.Fatalf("writing the run record printed: %q", live.String()[len(runOut):])
	}
	return runOut, runRecordPath(dir, gpuRunConfig()), src.asked
}

// metalRunEvents is each cell's stream, by cell name, with its tests taking act[cell name] ("pass" when unnamed).
func metalRunEvents(act map[string]string) map[string][]testEvent {
	evs := map[string][]testEvent{}
	cells := []cell{webgpuSuiteCell, webgpuParityCell}
	for _, s := range metalCellGroups("") {
		cells = append(cells, s.Cell)
	}
	for _, c := range cells {
		a := act[c.Name]
		if a == "" {
			a = "pass"
		}
		evs[c.Name] = metalCellEvents(metalCellTests(metalCellGroup{Cell: c}), a, false)
	}
	return evs
}

var (
	cgoClean = cgoFreeInput{Built: true, OtoolOK: true, Otool: "/tmp/gpu_gate_serve:\n\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n"}
	adapter  = webgpuProbeRecord{Present: true, Backend: "metal"}
)

// TestReplayGPU_equalsTheRun is SB12-C5's property at tier 0: cells graded and printed as the live gate does, recorded with
// writeGPURunRecord, replay to the same sections: the Metal groups, cgofree and group W. One cell is all-skip, so the
// runner's own note is replayed too, and lifecycle's checkpoint is present while its cell fails.
func TestReplayGPU_equalsTheRun(t *testing.T) {
	runOut, rec, _ := recordMetalRun(t, metalRunEvents(map[string]string{"metal-parity": "skip", "metal-lifecycle": "fail"}),
		map[string]int{"metal-lifecycle": 1}, cgoClean, adapter)
	var replay bytes.Buffer
	rc := replayGPU(&replay, rec)
	got, tail, ok := strings.Cut(replay.String(), "\nregraded ")
	if !ok {
		t.Fatalf("the replay printed no closing line:\n%s", replay.String())
	}
	if got != runOut {
		t.Errorf("the replay's sections differ from the run's:\n%s", lineDiff(runOut, got))
	}
	if rc != 1 || !strings.Contains(tail, "(2 fail") {
		t.Errorf("replay exit %d, closing line %q; want 1 and 2 failed groups (parity all-skip, lifecycle failed)", rc, tail)
	}
}

// TestReplayGPU_emptyOrAllSkipCellIsRed: the live gate fails a run whose filtered cell matched no test (EMPTY) or skipped
// every test (VACUOUS) through its tally, even where the group itself printed PASS, so the replay of such a run must not
// exit 0. A cell with no test events is empty; the suite cell is unfiltered, so an all-skip suite is neither, as live.
func TestReplayGPU_emptyOrAllSkipCellIsRed(t *testing.T) {
	evs := metalRunEvents(nil)
	evs["metal-lifecycle"] = []testEvent{{Action: "output", Package: metalPkg, Output: "ok  \t" + metalPkg + "\t0.100s [no tests to run]\n"}}
	_, rec, _ := recordMetalRun(t, evs, nil, cgoClean, adapter)
	var replay bytes.Buffer
	if rc := replayGPU(&replay, rec); rc != 1 || !strings.Contains(replay.String(), "1 empty cell(s)") {
		t.Errorf("an empty lifecycle cell replays with exit %d:\n%s\nwant exit 1 and the empty cell named", rc, replay.String())
	}
	_, rec, _ = recordMetalRun(t, metalRunEvents(nil), nil, cgoClean, adapter)
	replay.Reset()
	if rc := replayGPU(&replay, rec); rc != 0 {
		t.Errorf("a green run replays with exit %d:\n%s", rc, replay.String())
	}
}

// TestMetalCellGroups_guardsArePinned pins, by name, the Metal groups and the WebGPU parity cell that must fail on an all-skip cell (`go test` exits 0
// on one, so without the check such a group passes having verified nothing) and the groups that need the checkpoint.
// TestGradeMetalCell_plants reads the same flags from the specs, so it cannot see a flag removed; this test can. A list here
// only grows.
func TestMetalCellGroups_guardsArePinned(t *testing.T) {
	vacuousFails := []string{"parity", "lifecycle", "prefill"}
	needsModel := []string{"lifecycle", "prefill"}
	by := map[string]metalCellGroup{}
	for _, g := range metalCellGroups("/m/qwen.gguf") {
		by[g.Group] = g
	}
	for _, name := range vacuousFails {
		g, ok := by[name]
		if !ok {
			t.Errorf("no Metal cell group %q: it must fail on an all-skip cell", name)
			continue
		}
		if !g.VacuousFails {
			t.Errorf("%s no longer fails on an all-skip cell: an all-skip run of %s would print PASS", name, g.Cell.Run)
		}
		res, cr := gradeEvents(g.Cell, metalCellEvents(metalCellTests(g), "skip", false), 0)
		if v := gradeMetalCell(g, "/m/qwen.gguf", true, cr, res.text()); v.State != groupFail {
			t.Errorf("%s with every test skipped grades %d, want FAIL", name, v.State)
		}
	}
	res, cr := gradeEvents(webgpuParityCell, metalCellEvents(metalCellTests(metalCellGroup{Cell: webgpuParityCell}), "skip", false), 0)
	if v := gradeWebGPUParity(cr, res.text()); v.State != groupFail {
		t.Errorf("webgpu-parity with every test skipped grades %d, want FAIL: an all-skip run of %s would print PASS", v.State, webgpuParityCell.Run)
	}
	for _, name := range needsModel {
		g, ok := by[name]
		if !ok {
			t.Errorf("no Metal cell group %q: it needs the checkpoint", name)
			continue
		}
		if !g.NeedsModel {
			t.Errorf("%s no longer needs the checkpoint: without it the cell would run against nothing", name)
		}
		if v := gradeMetalCell(g, "/m/qwen.gguf", false, cellResult{}, ""); v.State != groupSkip {
			t.Errorf("%s with no checkpoint grades %d, want SKIP", name, v.State)
		}
	}
}

// TestGradeMetalCgoFree_plants is SB12-C4 for step 9's cgofree: a failed build fails; otool naming the Metal framework or
// MetalPerformanceShaders fails and lists the line; a clean linkage passes; otool not having run passes on the build alone.
func TestGradeMetalCgoFree_plants(t *testing.T) {
	linked := "/tmp/gpu_gate_serve:\n\t/System/Library/Frameworks/Metal.framework/Versions/A/Metal (compatibility version 1.0.0)\n"
	for name, tc := range map[string]struct {
		in   cgoFreeInput
		want groupState
		line string // a detail line the verdict must hold
	}{
		"clean":             {cgoClean, groupPass, ""},
		"build failed":      {cgoFreeInput{}, groupFail, ""},
		"links Metal":       {cgoFreeInput{Built: true, OtoolOK: true, Otool: linked}, groupFail, "        /System/Library/Frameworks/Metal.framework/Versions/A/Metal (compatibility version 1.0.0)"},
		"links MPS":         {cgoFreeInput{Built: true, OtoolOK: true, Otool: "\t/x/MetalPerformanceShaders.framework/MPS (c)\n"}, groupFail, "        /x/MetalPerformanceShaders.framework/MPS (c)"},
		"otool did not run": {cgoFreeInput{Built: true}, groupPass, ""},
	} {
		v := gradeMetalCgoFree(tc.in)
		if v.State != tc.want || (tc.line != "" && !slices.Contains(v.Lines, tc.line)) {
			t.Errorf("%s: state %d, lines %q; want state %d with %q", name, v.State, v.Lines, tc.want, tc.line)
		}
	}
}

// TestWebGPUGroup_plants is SB12-C4 for step 9's group W, through webgpuGroup on a fake source: the probe's opt-out and no
// adapter skip and a probe that never answered fails, with no cell run; a failed suite fails and runs no parity cell; an
// all-skip parity cell fails; a failed parity test fails naming its assertion; a green run passes both.
func TestWebGPUGroup_plants(t *testing.T) {
	run := func(probe webgpuProbeRecord, act map[string]string, rcs map[string]int) (*gpuGate, string, map[string]bool) {
		var out bytes.Buffer
		g := &gpuGate{w: &out, logDir: t.TempDir(), backend: "metal", emitted: map[string]bool{}, runIn: gpuRunInputs{WebGPU: &probe}}
		src := fakeLiveInputs{t: t, g: g, evs: metalRunEvents(act), rcs: rcs, asked: map[string]bool{}}
		if err := g.webgpuGroup(src); err != nil {
			t.Fatal(err)
		}
		return g, out.String(), src.asked
	}
	for name, p := range map[string]webgpuProbeRecord{"opt-out": {OptOut: true}, "no adapter": {}} {
		if g, out, asked := run(p, nil, nil); g.skipped != 1 || g.fail != 0 || len(asked) != 0 {
			t.Errorf("%s: skip %d fail %d, cells %v; want one SKIP and no cell\n%s", name, g.skipped, g.fail, asked, out)
		}
	}
	if g, out, asked := run(webgpuProbeRecord{Failed: "note: the probe never answered\n"}, nil, nil); g.fail != 1 || len(asked) != 0 {
		t.Errorf("a probe that never answered: fail %d, cells %v; want one FAIL and no cell\n%s", g.fail, asked, out)
	}
	g, out, asked := run(adapter, map[string]string{"webgpu-suite": "fail"}, map[string]int{"webgpu-suite": 1})
	if g.fail != 1 || asked["webgpu-parity"] || !strings.Contains(out, "x_test.go:12: got 1, want 2") {
		t.Errorf("a failed suite: fail %d, parity asked %v; want one FAIL with its assertion and no parity cell\n%s", g.fail, asked["webgpu-parity"], out)
	}
	if g, out, _ := run(adapter, map[string]string{"webgpu-parity": "skip"}, nil); g.fail != 1 || g.pass != 1 {
		t.Errorf("an all-skip parity cell: pass %d fail %d; want the suite PASS and parity FAIL\n%s", g.pass, g.fail, out)
	}
	g, out, _ = run(adapter, map[string]string{"webgpu-parity": "fail"}, map[string]int{"webgpu-parity": 1})
	if g.fail != 1 || !strings.Contains(out, "--- FAIL: ") || !strings.Contains(out, "x_test.go:12: got 1, want 2") {
		t.Errorf("a failed parity test: fail %d; want FAIL naming the test and its assertion\n%s", g.fail, out)
	}
	if g, out, _ := run(adapter, nil, nil); g.pass != 2 || g.fail != 0 || g.ran != 2 {
		t.Errorf("a green run: pass %d fail %d ran %d; want 2, 0, 2\n%s", g.pass, g.fail, g.ran, out)
	}
}

// TestReplayGPU_aMissingCellIsExit2: a record whose group W ran its cells but whose suite stream is not beside it cannot be
// regraded, and the replay says so with exit 2 instead of grading the groups it could.
func TestReplayGPU_aMissingCellIsExit2(t *testing.T) {
	_, rec, _ := recordMetalRun(t, metalRunEvents(nil), nil, cgoClean, adapter)
	if err := os.Remove(filepath.Join(filepath.Dir(rec), "gate_gpu_webgpu-suite.json")); err != nil {
		t.Fatal(err)
	}
	var replay bytes.Buffer
	if rc := replayGPU(&replay, rec); rc != 2 || !strings.Contains(replay.String(), "holds no webgpu-suite cell") {
		t.Errorf("a missing webgpu-suite stream replays with exit %d:\n%s", rc, replay.String())
	}
}

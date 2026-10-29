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
	clean cleanGPUInput
	repoI repoInput
	steps map[string]ciStepResult // by job and name; a row with none passes
	asked map[string]bool
}

func (f fakeLiveInputs) cell(c cell) (cellResult, *results, string, error) {
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
	return cr, res, res.text(), nil
}

func (f fakeLiveInputs) cleanGPU() (*cleanGPUInput, error) {
	in := f.clean
	f.g.runIn.CleanGPU = &in
	return &in, nil
}

func (f fakeLiveInputs) repo() (*repoInput, error) {
	in := f.repoI
	f.g.runIn.Repo = &in
	return &in, nil
}

func (f fakeLiveInputs) ciStep(_ int, r ciRow) (ciStepResult, error) {
	res, ok := f.steps[r.Job+"\x00"+r.Name]
	if !ok {
		res = ciStepResult{Job: r.Job, Name: r.Name, OK: true}
	}
	f.g.runIn.Repo.Steps = append(f.g.runIn.Repo.Steps, res)
	return res, nil
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

// TestGradeCleanGPU_plants is SB12-C4 for step 10's cleangpu: off CUDA and without nvidia-smi it skips; a quiet card passes;
// a card held by a stray serve process fails listing the processes; one held only by others passes listing them. Its
// passes do not count toward ran.
func TestGradeCleanGPU_plants(t *testing.T) {
	for name, tc := range map[string]struct {
		in   cleanGPUInput
		want groupState
		pre  int
	}{
		"metal":         {cleanGPUInput{Backend: "metal"}, groupSkip, 0},
		"no nvidia-smi": {cleanGPUInput{Backend: "cuda"}, groupSkip, 0},
		"quiet":         {cleanGPUInput{Backend: "cuda", HaveSMI: true, Used: "5"}, groupPass, 0},
		"stray serve":   {cleanGPUInput{Backend: "cuda", HaveSMI: true, Procs: "4242, 812 MiB, /home/f/bin/goinfer-serve", Used: "900"}, groupFail, 2},
		"others only":   {cleanGPUInput{Backend: "cuda", HaveSMI: true, Procs: "77, 16 MiB, /usr/bin/kwin_wayland", Used: "16"}, groupPass, 2},
	} {
		v := gradeCleanGPU(tc.in)
		if v.State != tc.want || len(v.Pre) != tc.pre || (v.State == groupPass && !v.Uncounted) {
			t.Errorf("%s: state %d, %d process lines, uncounted %v; want state %d and %d lines", name, v.State, len(v.Pre), v.Uncounted, tc.want, tc.pre)
		}
	}
}

// TestGradeSeam_plants is SB12-C4 for step 10's seam: a non-zero exit fails with detail, a run that matched no test fails,
// and a green one passes and counts toward ran.
func TestGradeSeam_plants(t *testing.T) {
	res, cr := gradeEvents(seamCell, metalCellEvents([]string{"TestSeam_a"}, "pass", false), 0)
	if v := gradeSeam(cr, res.ranCount(), res.text()); v.State != groupPass || v.Uncounted {
		t.Errorf("a green seam: state %d uncounted %v", v.State, v.Uncounted)
	}
	res, cr = gradeEvents(seamCell, metalCellEvents([]string{"TestSeam_a"}, "fail", false), 1)
	if v := gradeSeam(cr, res.ranCount(), res.text()); v.State != groupFail || !strings.Contains(strings.Join(v.Lines, "\n"), "--- FAIL: TestSeam_a") {
		t.Errorf("a failed seam test: state %d, lines %q", v.State, v.Lines)
	}
	res, cr = gradeEvents(seamCell, []testEvent{{Action: "output", Package: metalPkg, Output: "ok  \t" + metalPkg + "\t0.1s [no tests to run]\n"}}, 0)
	if v := gradeSeam(cr, res.ranCount(), res.text()); v.State != groupFail || !strings.Contains(v.Line, "matched NO tests") {
		t.Errorf("a seam run that matched no test: state %d line %q, want FAIL", v.State, v.Line)
	}
}

// repoGreen is a repo group's inputs on a Darwin host: a green lint, two darwin steps (one runner-only), one linux step.
var repoGreen = repoInput{LintOK: true, LintOut: "queue_citation_lint: 12 checks\nGITIGNORED DESTINATIONS: none.\n", RowsOK: true, Darwin: true,
	Rows: "root-darwin\tgofmt\tstep\t-\tgofmt -l .\nmetal-darwin\tdevice tests\trunner:a real Metal GPU\t-\tgo test ./metal/\nroot\tvet\tstep\t-\tgo vet ./...\n"}

// TestRepoGroup_plants is SB12-C4 for step 10's repo, through repoGroup on a fake source: a green run passes the lint and
// one reproduced step, skips the runner-only step and the other platform's one; a red lint fails with its lines; a failed
// derivation fails and runs no step; a failed step fails with its lines and drops the closing pass.
func TestRepoGroup_plants(t *testing.T) {
	run := func(in repoInput, steps map[string]ciStepResult) (*gpuGate, string) {
		var out bytes.Buffer
		g := &gpuGate{w: &out, emitted: map[string]bool{}}
		if err := g.repoGroup(fakeLiveInputs{t: t, g: g, repoI: in, steps: steps, asked: map[string]bool{}}); err != nil {
			t.Fatal(err)
		}
		return g, out.String()
	}
	g, out := run(repoGreen, nil)
	if g.pass != 2 || g.skipped != 2 || g.fail != 0 || g.ran != 0 || len(g.runIn.Repo.Steps) != 1 ||
		!strings.Contains(out, "GITIGNORED DESTINATIONS: none.") || !strings.Contains(out, "1 CI hygiene check(s) reproduced") ||
		!strings.Contains(out, "1 linux-only CI hygiene step(s)") {
		t.Errorf("a green repo group: pass %d skip %d fail %d ran %d, %d steps run\n%s", g.pass, g.skipped, g.fail, g.ran, len(g.runIn.Repo.Steps), out)
	}
	red := repoGreen
	red.LintOK, red.LintOut = false, "RED: docs/x.md cites a7f3 which does not resolve\nline 2\n"
	if g, out := run(red, nil); g.fail != 1 || !strings.Contains(out, "      RED: docs/x.md cites a7f3") {
		t.Errorf("a red lint: fail %d\n%s", g.fail, out)
	}
	none := repoGreen
	none.RowsOK, none.Rows, none.RowsErr = false, "", "ci.yml: no such file"
	if g, out := run(none, nil); g.fail != 1 || g.runIn.Repo.Steps != nil || !strings.Contains(out, "cannot derive CI's check set: ci.yml: no such file") {
		t.Errorf("a failed derivation: fail %d, steps %v\n%s", g.fail, g.runIn.Repo.Steps, out)
	}
	g, out = run(repoGreen, map[string]ciStepResult{"root-darwin\x00gofmt": {Job: "root-darwin", Name: "gofmt", Out: "a.go\nb.go\n"}})
	if g.fail != 1 || !strings.Contains(out, "CI[root-darwin] gofmt") || !strings.Contains(out, "      a.go") || strings.Contains(out, "reproduced locally") {
		t.Errorf("a failed step: fail %d\n%s", g.fail, out)
	}
}

// fullMetalRun runs a whole Metal gate through runGPU's sequence on a fake source, with fixed verdict inputs, writes the
// run record, and returns what it printed from the clean-GPU check through the verdict, its exit code and the record.
func fullMetalRun(t *testing.T, steps map[string]ciStepResult) (string, int, string) {
	t.Helper()
	dir := t.TempDir()
	var live bytes.Buffer
	model := filepath.Join(dir, "qwen.gguf")
	probe := adapter
	g := &gpuGate{w: &live, logDir: dir, backend: "metal", emitted: map[string]bool{}, expect: gpuExpect("metal"),
		commit: "0123456789abcdef", notes: []string{"a note the runner made"},
		runIn: gpuRunInputs{MetalModel: model, MetalModelPresent: true, WebGPU: &probe}}
	g.runIn.RunnerNotes = slices.Clone(g.notes)
	src := fakeLiveInputs{t: t, g: g, evs: metalRunEvents(nil), cgo: cgoClean, clean: cleanGPUInput{Backend: "metal"},
		repoI: repoGreen, steps: steps, asked: map[string]bool{}}
	src.evs["seam"] = metalCellEvents([]string{"TestSeam_a"}, "pass", false)
	for _, step := range []func(gpuInputs) error{g.cleanGPUGroup, g.seamGroup, g.metalGroups, g.webgpuGroup, g.repoGroup} {
		if err := step(src); err != nil {
			t.Fatal(err)
		}
	}
	g.runIn.Verdict = &gpuVerdictInputs{Backend: "metal", Commit: g.commit, Host: "Darwin", Now: "2026-10-29T22:00:00Z"}
	n := live.Len()
	g.writeGPURunRecord()
	if live.Len() != n {
		t.Fatalf("writing the run record printed: %q", live.String()[n:])
	}
	code := g.verdict()
	return live.String(), code, runRecordPath(dir, gpuRunConfig())
}

// TestReplayGPU_aFullRecordReplaysTheVerdict is SB12-C5's property at tier 0 for step 10: a Metal record that holds every
// group's inputs and the verdict's replays to the run's whole output, the verdict block included, and to its exit code, both
// when it passes and when a CI step fails.
func TestReplayGPU_aFullRecordReplaysTheVerdict(t *testing.T) {
	for name, tc := range map[string]struct {
		steps map[string]ciStepResult
		code  int
	}{
		"pass":          {nil, 0},
		"a step failed": {map[string]ciStepResult{"root-darwin\x00gofmt": {Job: "root-darwin", Name: "gofmt", Out: "a.go\n"}}, 1},
	} {
		live, code, rec := fullMetalRun(t, tc.steps)
		if code != tc.code {
			t.Errorf("%s: the run exits %d, want %d\n%s", name, code, tc.code, live)
		}
		var replay bytes.Buffer
		rc := replayGPU(&replay, rec)
		if replay.String() != live {
			t.Errorf("%s: the replay differs from the run:\n%s", name, lineDiff(live, replay.String()))
		}
		if rc != code || !strings.Contains(live, "== verdict ==") {
			t.Errorf("%s: the replay exits %d, the run %d", name, rc, code)
		}
	}
}

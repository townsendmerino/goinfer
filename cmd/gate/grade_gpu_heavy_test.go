package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestGradeHeavy_plants is SB12-C4 for step 12 (docs/tasks/task-subsystems-2026-10.md, SB12 step 12 (CUDA)): each part of the
// heavy tier can go red on its own input.
func TestGradeHeavy_plants(t *testing.T) {
	green := heavyInput{Drain: []string{"TestD"}, Iso: []string{"TestI"}, Total: 3, Kernels: 1,
		Listed: []string{"TestA", "TestD", "TestI", "TestK"},
		Groups: []heavyGroup{{Label: "small", Tests: []string{"TestA", "TestD", "TestI"}}, {Label: "kernels", Tests: []string{"TestK"}}}}
	vs, run := gradeHeavyStart(green)
	if !run || len(vs) != 1 || vs[0].State != groupNone || !strings.Contains(strings.Join(vs[0].Lines, "\n"), "small          3 test(s)") {
		t.Fatalf("green start: run %v %+v", run, vs)
	}
	if vs, run := gradeHeavyStart(heavyInput{Skipped: true}); run || vs[0].State != groupSkip {
		t.Errorf("a skipped tier: run %v %+v", run, vs)
	}
	starts := map[string]heavyInput{
		"an empty drain derivation": func() heavyInput { in := green; in.Drain = nil; return in }(),
		"a failed listing":          {Drain: green.Drain, ListErr: "exit status 1"},
		"a failed partition":        {Drain: green.Drain, Total: 3, PartitionErr: "cannot read cuda/x_test.go"},
		"a failed kernels listing":  {Drain: green.Drain, Total: 3, KernelsErr: "exit status 2"},
		"a test in no group":        func() heavyInput { in := green; in.Problems = []string{"TestOrphan: names no family"}; return in }(),
	}
	for name, in := range starts {
		vs, _ := gradeHeavyStart(in)
		failed := false
		for _, v := range vs {
			failed = failed || v.State == groupFail
		}
		if !failed {
			t.Errorf("%s: no FAIL in %+v", name, vs)
		}
	}

	h := green.Groups[0]
	lines, notes, rec := heavyProcLines(h, cellResult{Skip: 3}, procPeak{})
	if j := strings.Join(lines, "\n"); !strings.Contains(j, "NOT SAMPLED") || !strings.Contains(j, "all 3 test(s) SKIPPED") || len(notes) != 1 || rec.Tests != 3 {
		t.Errorf("a process that ran nothing: lines %q notes %q rec %+v", lines, notes, rec)
	}
	lines, _, rec = heavyProcLines(h, cellResult{RC: 1, Pass: 1, LogPath: "/l/x.json", Dur: 61 * time.Second},
		procPeak{PID: 7, HWMKB: 2048 * 1024, Samples: 4, ByTest: map[string]int64{"TestA": 1024 * 1024}})
	if j := strings.Join(lines, "\n"); !strings.Contains(j, "VmHWM 2048 MB") || !strings.Contains(j, "TestA 1024 MB") ||
		!strings.Contains(j, "small full output: /l/x.json") || rec.WallS != 61 || rec.VmHWMMB != 2048 {
		t.Errorf("a failed sampled process: lines %q rec %+v", lines, rec)
	}

	if v := gradeHeavySplit(4, nil, nil); v.State != groupNone {
		t.Errorf("a clean split: %+v", v)
	}
	for _, c := range [][2][]string{{{"TestA"}, nil}, {nil, {"TestD"}}} {
		if v := gradeHeavySplit(4, c[0], c[1]); v.State != groupFail || len(v.Notes) != 1 {
			t.Errorf("a split missing %v, doubled %v: %+v", c[0], c[1], v)
		}
	}

	failed := metalCellEvents([]string{"TestD"}, "fail", false)
	res, cr := gradeEvents(cell{Name: "cuda-drain"}, failed, 1)
	for _, c := range []struct {
		name                      string
		skipped, ran, rc, wantBad int
	}{{"green", 1, 1, 0, 0}, {"skipped short", 0, 1, 0, 1}, {"ran short", 1, 0, 0, 1}, {"failed", 1, 1, 1, 1}} {
		dcr := cellResult{RC: c.rc, LogPath: "/l/drain.json"}
		out := ""
		if c.rc != 0 {
			dcr, out = cr, res.text()
		}
		bad := 0
		for _, v := range gradeHeavyDrain(1, c.skipped, c.ran, dcr, out) {
			if v.State == groupFail {
				bad++
				if c.rc != 0 && !strings.Contains(strings.Join(v.Lines, "\n"), "--- FAIL: TestD") {
					t.Errorf("a failed drain test names no failure: %q", v.Lines)
				}
			}
		}
		if bad != c.wantBad {
			t.Errorf("drain %s: %d failure(s), want %d", c.name, bad, c.wantBad)
		}
	}

	if _, bad := gradeHeavyIsoTest("TestI", cellResult{}, ""); bad {
		t.Error("a fresh-process test that passed is reported")
	}
	if v, bad := gradeHeavyIsoTest("TestI", cr, res.text()); !bad || v.State != groupFail || v.Line != "TestI (fresh process)" {
		t.Errorf("a failed fresh-process test: %+v", v)
	}
	if v := gradeHeavyIso(1, 1, 1, 0); v.State != groupPass {
		t.Errorf("reconciled fresh-process tests: %+v", v)
	}
	for _, c := range [][3]int{{0, 1, 0}, {1, 0, 0}} {
		if v := gradeHeavyIso(1, c[0], c[1], c[2]); v.State != groupFail {
			t.Errorf("fresh-process counts %v: %+v", c, v)
		}
	}
	if v := gradeHeavyIso(1, 1, 1, 1); v.State != groupNone {
		t.Errorf("reconciled but one failed: %+v, want lines only (the failure was its own verdict)", v)
	}

	if v := gradeHeavyTier(1, 10, res.text(), 2, nil); v.State != groupFail || !strings.Contains(strings.Join(v.Lines, "\n"), "--- FAIL: TestD") {
		t.Errorf("a failed main group: %+v", v)
	}
	v := gradeHeavyTier(0, 10, "", 3, [][]string{{"TestA"}, nil, {"TestK"}})
	if j := strings.Join(v.Lines, "\n"); v.State != groupPass || !strings.Contains(j, "ran 3 tests, skipped 2") || !strings.Contains(j, "· TestA") ||
		!strings.Contains(j, "· TestK") || len(v.Notes) != 1 {
		t.Errorf("a green tier with skips: %+v", v)
	}
}

// fakeHeavyLive is a live source for the heavy tier at tier 0: each process's stream is written where runCell writes it and its
// results print as they arrive, as gpuGate.run and runHeavyGroup print them.
type fakeHeavyLive struct {
	fakeLiveInputs
	peaks []procPeak
}

func (f fakeHeavyLive) streamed(c cell) (cellResult, *results, string) {
	path := filepath.Join(f.g.logDir, "gate_"+sanitize("gpu_"+c.Name)+".json")
	var b bytes.Buffer
	for _, ev := range f.evs[c.Name] {
		j, _ := json.Marshal(ev)
		b.Write(append(j, '\n'))
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		f.t.Fatal(err)
	}
	res := newResults()
	res.cur = c.Name
	res.stream = func(_, test, action string) {
		f.g.w.Write([]byte("        · " + strings.ToUpper(action) + ": " + test + "\n"))
	}
	for _, ev := range f.evs[c.Name] {
		res.add(ev)
	}
	cr := gradeCell(c, gpuRunConfig(), res, f.rcs[c.Name], path, 0)
	f.g.cells = append(f.g.cells, cr)
	return cr, res, res.text()
}

func (f fakeHeavyLive) heavyCell(i int, h heavyGroup) (cellResult, *results, string, procPeak, error) {
	c := cell{Name: "cuda-heavy-" + h.Label, Run: h.Run}
	cr, res, out := f.streamed(c)
	f.g.noteIfEmpty(c, res)
	f.g.runIn.Heavy.Peaks = append(f.g.runIn.Heavy.Peaks, f.peaks[i])
	return cr, res, out, f.peaks[i], nil
}

func (f fakeHeavyLive) streamCell(c cell) (cellResult, *results, string, error) {
	cr, res, out := f.streamed(c)
	f.g.noteIfEmpty(c, res)
	f.g.noteIfAllSkipped(c, cr)
	return cr, res, out, nil
}

// TestReplayGPU_heavyEqualsTheRun is SB12-C5's property for step 12 at tier 0: a fake heavy tier (two main-group processes, one
// with a failed test, the drain group and one fresh-process test, each with the markers' tokens) run through heavyTier,
// recorded with writeGPURunRecord and replayed, prints the same section, streamed results and memory lines included.
func TestReplayGPU_heavyEqualsTheRun(t *testing.T) {
	dir := t.TempDir()
	var live bytes.Buffer
	in := heavyInput{Drain: []string{"TestD"}, Iso: []string{"TestI"}, Total: 3, Kernels: 1, Secs: 42,
		Listed: []string{"TestA", "TestD", "TestI", "TestK"},
		Groups: []heavyGroup{{Label: "small", Tests: []string{"TestA", "TestD", "TestI"}}, {Label: "kernels", Tests: []string{"TestK"}}}}
	g := &gpuGate{w: &live, logDir: dir, backend: "cuda", emitted: map[string]bool{}, runIn: gpuRunInputs{Heavy: &in}}
	marker := func(evs []testEvent, test, token string) []testEvent {
		return append(evs[:1], append([]testEvent{{Action: "output", Package: metalPkg, Test: test, Output: "    m.go:1: " + token + "\n"}}, evs[1:]...)...)
	}
	small := metalCellEvents([]string{"TestA"}, "fail", false)
	small = append(small[:len(small)-1], append(marker(metalCellEvents([]string{"TestD"}, "skip", false), "TestD", "DRAIN-GROUP-SKIP"),
		marker(metalCellEvents([]string{"TestI"}, "skip", false), "TestI", "ISOLATED-SKIP")...)...)
	evs := map[string][]testEvent{
		"cuda-heavy-small":   small,
		"cuda-heavy-kernels": metalCellEvents([]string{"TestK"}, "pass", false),
		"cuda-drain":         marker(metalCellEvents([]string{"TestD"}, "pass", false), "TestD", "DRAIN-GROUP-RUN"),
		"cuda-iso-TestI":     marker(metalCellEvents([]string{"TestI"}, "pass", false), "TestI", "ISOLATED-RUN"),
	}
	src := fakeHeavyLive{fakeLiveInputs: fakeLiveInputs{t: t, g: g, evs: evs, rcs: map[string]int{"cuda-heavy-small": 1}, asked: map[string]bool{}},
		peaks: []procPeak{{PID: 9, HWMKB: 4 << 20, Samples: 3, ByTest: map[string]int64{"TestA": 3 << 20}}, {}}}
	g.runIn.Heavy = &heavyInput{}
	*g.runIn.Heavy = in
	g.runIn.Heavy.Peaks = nil
	if err := g.heavyTier(src); err != nil {
		t.Fatal(err)
	}
	runOut := live.String()
	g.writeGPURunRecord()
	for _, want := range []string{"· FAIL: TestA", "· PASS: TestK", "VmHWM 4096 MB", "ran-in-drain=1", "ran-alone=1", "— 42s"} {
		if !strings.Contains(runOut, want) {
			t.Errorf("the run printed no %q:\n%s", want, runOut)
		}
	}
	rec, cells, texts, err := loadGPURunRecord(runRecordPath(dir, gpuRunConfig()), func(string) bool { return true })
	if err != nil {
		t.Fatal(err)
	}
	var replay bytes.Buffer
	rg := &gpuGate{w: &replay, backend: "cuda", emitted: map[string]bool{}, runIn: *rec.GPU}
	if err := rg.heavyTier(&recordedInputs{g: rg, path: runRecordPath(dir, gpuRunConfig()), cells: cells, texts: texts}); err != nil {
		t.Fatal(err)
	}
	if replay.String() != runOut {
		t.Errorf("the replay's heavy section differs from the run's:\n%s", lineDiff(runOut, replay.String()))
	}
	if rg.fail != g.fail || rg.pass != g.pass || rg.ran != g.ran || strings.Join(rg.notes, "|") != strings.Join(g.notes, "|") ||
		len(rg.heavyProcs) != 2 || rg.heavyProcs[0] != g.heavyProcs[0] {
		t.Errorf("the replay's tally differs: fail %d/%d pass %d/%d ran %d/%d notes %q/%q procs %+v/%+v",
			rg.fail, g.fail, rg.pass, g.pass, rg.ran, g.ran, rg.notes, g.notes, rg.heavyProcs, g.heavyProcs)
	}
	if g.fail != 1 {
		t.Errorf("the run failed %d verdict(s), want 1 (the main group)", g.fail)
	}
}

// TestHeavyNarrowed is SB12-C7 for step 12: -heavy-group runs only the named groups and the drain and fresh-process tests inside
// them, says so in the partition block and as a note, and a name that is no heavy group refuses the run before it starts.
func TestHeavyNarrowed(t *testing.T) {
	in := heavyInput{Drain: []string{"TestD1", "TestD2"}, Iso: []string{"TestI1", "TestI2"}, Total: 4, Kernels: 1,
		Listed: []string{"TestA", "TestD1", "TestI1", "TestB", "TestD2", "TestI2", "TestK"},
		Groups: []heavyGroup{{Label: "small", Tests: []string{"TestA", "TestD1", "TestI1"}},
			{Label: "qwen3-4b", Tests: []string{"TestB", "TestD2", "TestI2"}}, {Label: "kernels", Tests: []string{"TestK"}}},
		Run: []string{"qwen3-4b", "kernels"}}
	groups, listed, drain, iso := heavyRunSet(in)
	if len(groups) != 2 || groups[0].Label != "qwen3-4b" || strings.Join(listed, " ") != "TestB TestD2 TestI2 TestK" ||
		strings.Join(drain, " ") != "TestD2" || strings.Join(iso, " ") != "TestI2" {
		t.Errorf("narrowed set: groups %v listed %v drain %v iso %v", groups, listed, drain, iso)
	}
	vs, run := gradeHeavyStart(in)
	if !run || !strings.Contains(strings.Join(vs[len(vs)-1].Lines, "\n"), "NARROWED by -heavy-group to qwen3-4b, kernels: 2 of 3 process(es), with 1 drain and 1 fresh-process") ||
		len(vs[len(vs)-1].Notes) != 1 || !strings.Contains(vs[len(vs)-1].Notes[0], "not a release verdict") {
		t.Errorf("a narrowed start: %+v", vs)
	}
	in.Run = nil
	if g, l, d, i := heavyRunSet(in); len(g) != 3 || len(l) != 7 || len(d) != 2 || len(i) != 2 {
		t.Errorf("an unnarrowed run set: %d %d %d %d", len(g), len(l), len(d), len(i))
	}
	var out bytes.Buffer
	if rc := runGPU(&out, t.TempDir(), "", []string{"qwen3-4b", "no-such-group"}); rc != 2 || !strings.Contains(out.String(), `no heavy group "no-such-group"`) {
		t.Errorf("an unknown -heavy-group name: exit %d\n%s", rc, out.String())
	}
}

// TestHeavyNarrowed_noDrainInside: a run narrowed to groups with no drain test runs no drain process (a -run of no tests is an
// empty cell, which fails the gate) and says so; the same run's replay prints the same.
func TestHeavyNarrowed_noDrainInside(t *testing.T) {
	dir := t.TempDir()
	var live bytes.Buffer
	in := heavyInput{Drain: []string{"TestD"}, Total: 2, Kernels: 1, Listed: []string{"TestA", "TestD", "TestK"},
		Groups: []heavyGroup{{Label: "small", Tests: []string{"TestA", "TestD"}}, {Label: "kernels", Tests: []string{"TestK"}}},
		Run:    []string{"kernels"}, Secs: 5}
	g := &gpuGate{w: &live, logDir: dir, backend: "cuda", emitted: map[string]bool{}, runIn: gpuRunInputs{Heavy: &in}}
	src := fakeHeavyLive{fakeLiveInputs: fakeLiveInputs{t: t, g: g, asked: map[string]bool{},
		evs: map[string][]testEvent{"cuda-heavy-kernels": metalCellEvents([]string{"TestK"}, "pass", false)}}, peaks: []procPeak{{}}}
	if err := g.heavyTier(src); err != nil {
		t.Fatal(err)
	}
	for _, c := range g.cells {
		if c.Cell.Name == "cuda-drain" {
			t.Errorf("a narrowed run with no drain test inside ran a drain cell")
		}
	}
	if len(g.emptyCells) > 0 || g.fail > 0 || !strings.Contains(live.String(), "no drain test inside the narrowed groups") {
		t.Errorf("empty cells %v, %d failure(s), output:\n%s", g.emptyCells, g.fail, live.String())
	}
	if _, none := heavyNoDrain(nil, nil); none {
		t.Error("an unnarrowed run with no drain test skips the drain process; gradeHeavyStart must fail it instead")
	}
}

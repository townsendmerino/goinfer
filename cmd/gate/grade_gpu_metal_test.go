package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
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
		if v := gradeMetalCell(g, model, true, cr, res.text()); v.State != groupFail || v.Line != g.Bad || len(v.Lines) == 0 {
			t.Errorf("%s with a failed test: state %d line %q, %d detail lines; want FAIL with detail", g.Group, v.State, v.Line, len(v.Lines))
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

// TestReplayGPU_equalsTheRun is SB12-C5's property at tier 0: cells graded and printed as the live gate does, recorded with
// writeGPURunRecord, replay to the same four sections. One cell is all-skip, so the runner's own note is replayed too, and
// lifecycle's checkpoint is present while its cell fails.
func TestReplayGPU_equalsTheRun(t *testing.T) {
	dir := t.TempDir()
	var live bytes.Buffer
	model := filepath.Join(dir, "qwen.gguf")
	g := &gpuGate{w: &live, logDir: dir, backend: "metal", emitted: map[string]bool{},
		runIn: gpuRunInputs{MetalModel: model, MetalModelPresent: true}}
	acts := map[string]string{"suite": "pass", "parity": "skip", "lifecycle": "fail", "prefill": "pass"}
	for _, s := range metalCellGroups(model) {
		g.grp(s.Group)
		g.hdr(s.Header)
		evs := metalCellEvents(metalCellTests(s), acts[s.Group], false)
		path := filepath.Join(dir, "gate_"+sanitize("gpu_"+s.Cell.Name)+".json")
		var b bytes.Buffer
		for _, ev := range evs {
			j, _ := json.Marshal(ev)
			b.Write(append(j, '\n'))
		}
		if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		rc := map[string]int{"fail": 1}[acts[s.Group]]
		res, cr := gradeEvents(s.Cell, evs, rc)
		cr.LogPath = path
		g.cells = append(g.cells, cr)
		g.noteIfEmpty(s.Cell, res)
		g.noteIfAllSkipped(s.Cell, cr)
		g.apply(gradeMetalCell(s, model, true, cr, res.text()))
	}
	runOut := live.String()
	g.writeGPURunRecord()
	if live.Len() != len(runOut) {
		t.Fatalf("writing the run record printed: %q", live.String()[len(runOut):])
	}
	var replay bytes.Buffer
	rc := replayGPU(&replay, runRecordPath(dir, gpuRunConfig()))
	got, tail, ok := strings.Cut(replay.String(), "\nregraded ")
	if !ok {
		t.Fatalf("the replay printed no closing line:\n%s", replay.String())
	}
	if got != runOut {
		t.Errorf("the replay's sections differ from the run's:\n%s", lineDiff(runOut, got))
	}
	if rc != 1 || !strings.Contains(tail, "(2 fail)") {
		t.Errorf("replay exit %d, closing line %q; want 1 and 2 failed groups (parity all-skip, lifecycle failed)", rc, tail)
	}
}

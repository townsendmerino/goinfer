package main

import (
	"strings"
	"testing"
)

// TestGradeCUDACells_plants is SB12-C4 for step 10 (docs/tasks/task-subsystems-2026-10.md, SB12 step 10 (CUDA)): each CUDA
// cell group fails on a failed test, naming its assertion, and on a crash, with the crash head; the suite lists its skips by
// name and warns on a census of zero; parity prints the all-zero-buffer note with the recorded reading; graphsforced fails on
// a cell that ran nothing, and always adds its forced-capture note.
func TestGradeCUDACells_plants(t *testing.T) {
	tests := []string{"TestA", "TestB"}
	join := func(v groupVerdict) string { return strings.Join(v.Lines, "\n") }
	assertFails := func(name string, v groupVerdict) {
		t.Helper()
		d := join(v)
		if v.State != groupFail || !strings.Contains(d, "--- FAIL: TestA") || !strings.Contains(d, "x_test.go:12: got 1, want 2") ||
			strings.Contains(d, "NO ASSERTION LINE MATCHED") {
			t.Errorf("%s with a failed test: state %d, detail:\n%s\nwant FAIL naming the failed test and its assertion", name, v.State, d)
		}
	}
	assertCrash := func(name string, v groupVerdict) {
		t.Helper()
		if v.State != groupFail || len(v.Lines) == 0 || !strings.Contains(v.Lines[0], "PROCESS DIED") {
			t.Errorf("%s with a crash: state %d, lines %q; want FAIL with the crash head", name, v.State, v.Lines)
		}
	}

	// suite
	res, cr := gradeEvents(cudaSuiteCell, metalCellEvents(tests, "fail", false), 1)
	assertFails("suite", gradeCUDASuite(cr, nil, res.text()))
	res, cr = gradeEvents(cudaSuiteCell, metalCellEvents(tests, "pass", true), 2)
	assertCrash("suite", gradeCUDASuite(cr, nil, res.text()))
	res, cr = gradeEvents(cudaSuiteCell, metalCellEvents(tests, "pass", false), 0)
	v := gradeCUDASuite(cr, []string{"TestHeavyOne", "TestHeavyTwo"}, res.text())
	if d := join(v); v.State != groupPass || !strings.Contains(d, "skipped within it: 2") || !strings.Contains(d, "· TestHeavyOne") ||
		!strings.Contains(d, "· TestHeavyTwo") {
		t.Errorf("suite green with two skips: state %d, lines:\n%s\nwant PASS listing both skips by name", v.State, d)
	}
	if v := gradeCUDASuite(cr, nil, res.text()); v.State != groupPass || !strings.Contains(join(v), "skip census: 0") {
		t.Errorf("suite green with no skips: state %d, lines:\n%s\nwant PASS with the zero-census warning", v.State, join(v))
	}
	res, cr = gradeEvents(cudaSuiteCell, metalCellEvents(tests, "fail", false), 1)
	if v := gradeCUDASuite(cr, []string{"TestHeavyOne"}, res.text()); !strings.Contains(join(v), "· TestHeavyOne") {
		t.Errorf("a failed suite drops its skip census:\n%s", join(v))
	}

	// parity
	res, cr = gradeEvents(cudaParityCell, metalCellEvents(tests, "fail", false), 1)
	assertFails("parity", gradeCUDAParity(cr, res.text(), ""))
	res, cr = gradeEvents(cudaParityCell, metalCellEvents(tests, "pass", true), 2)
	assertCrash("parity", gradeCUDAParity(cr, res.text(), ""))
	res, cr = gradeEvents(cudaParityCell, metalCellEvents(tests, "pass", false), 0)
	if v := gradeCUDAParity(cr, res.text(), ""); v.State != groupPass || len(v.Lines) == 0 {
		t.Errorf("parity green: state %d, %d evidence lines; want PASS with evidence", v.State, len(v.Lines))
	}
	if v := gradeCUDAParity(cellResult{RC: 1}, "    x_test.go:9: cosine 0.000000\n", "123 MiB"); !strings.Contains(join(v), "right now: 123 MiB") {
		t.Errorf("parity with a zero cosine does not print the recorded reading:\n%s", join(v))
	}

	// graphsforced
	graphs := []string{"TestGemma4Graphs_a", "TestGemma4Graphs_b"}
	res, cr = gradeEvents(cudaGraphsCell, metalCellEvents(graphs, "fail", false), 1)
	v = gradeCUDAGraphs(cr, res.ranCount(), nil, res.text())
	if d := join(v); v.State != groupFail || !strings.Contains(d, "x_test.go:12: got 1, want 2") {
		t.Errorf("graphs with a failed test: state %d, detail:\n%s", v.State, d)
	}
	res, cr = gradeEvents(cudaGraphsCell, metalCellEvents(graphs, "pass", true), 2)
	if v := gradeCUDAGraphs(cr, res.ranCount(), nil, res.text()); v.State != groupFail || !strings.Contains(v.Lines[0], "PROCESS DIED") {
		t.Errorf("graphs with a crash: state %d, lines %q", v.State, v.Lines)
	}
	if v := gradeCUDAGraphs(cellResult{}, 0, nil, ""); v.State != groupFail || !strings.Contains(v.Line, "matched NO tests") {
		t.Errorf("graphs that ran nothing: state %d line %q, want FAIL", v.State, v.Line)
	}
	res, cr = gradeEvents(cudaGraphsCell, metalCellEvents(graphs, "pass", false), 0)
	v = gradeCUDAGraphs(cr, res.ranCount(), []string{"TestGemma4Graphs_c"}, res.text())
	if v.State != groupPass || !strings.Contains(v.Line, "(2 tests, 1 skipped)") || !strings.Contains(join(v), "· TestGemma4Graphs_c") {
		t.Errorf("graphs green: state %d line %q lines %q", v.State, v.Line, v.Lines)
	}
	for _, v := range []groupVerdict{v, gradeCUDAGraphs(cellResult{}, 0, nil, "")} {
		if len(v.Notes) != 1 || !strings.Contains(v.Notes[0], "graphs are FORCED") {
			t.Errorf("graphs state %d: notes %q, want the forced-capture note", v.State, v.Notes)
		}
	}
}

// TestGradeCUDACgoFree_plants is SB12-C4 for step 10's cgofree: a failed build fails with the build's lines; ldd naming a CUDA
// library fails, listing the line, and still counts as ran; a clean linkage passes; ldd not run passes on the build alone.
func TestGradeCUDACgoFree_plants(t *testing.T) {
	v := gradeCUDACgoFree(cgoFreeInput{BuildOut: "# github.com/townsendmerino/goinfer/cuda/cmd/serve\nmain.go:3:2: undefined: x\n"})
	if v.State != groupFail || !strings.Contains(strings.Join(v.Lines, "\n"), "undefined: x") {
		t.Errorf("a failed build: state %d lines %q, want FAIL with the build's line", v.State, v.Lines)
	}
	for _, lib := range []string{"libcuda.so.1", "libnvrtc.so.12", "libcudart.so.12"} {
		ldd := "\tlinux-vdso.so.1 (0x0)\n\t" + lib + " => /usr/lib64/" + lib + " (0x1)\n"
		v := gradeCUDACgoFree(cgoFreeInput{Built: true, LddOK: true, Ldd: ldd})
		if v.State != groupFail || !v.Ran || !strings.Contains(strings.Join(v.Lines, "\n"), lib) {
			t.Errorf("a binary linking %s: state %d ran %v lines %q, want FAIL listing it, counted as ran", lib, v.State, v.Ran, v.Lines)
		}
	}
	if v := gradeCUDACgoFree(cgoFreeInput{Built: true, LddOK: true, Ldd: "\tnot a dynamic executable\n"}); v.State != groupPass {
		t.Errorf("a clean linkage: state %d, want PASS", v.State)
	}
	if v := gradeCUDACgoFree(cgoFreeInput{Built: true}); v.State != groupPass {
		t.Errorf("ldd not run: state %d, want PASS on the build alone", v.State)
	}
}

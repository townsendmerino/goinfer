package main

import (
	"fmt"
	"regexp"
	"strings"
)

// The gpu gate's cgofree (Metal) and webgpu groups as graders (SB12 step 9, docs/tasks/task-subsystems-2026-10.md): each
// verdict is a function of what the run recorded, the build and otool's output for cgofree, the adapter probe and the cells
// for webgpu.

// cgoFreeInput is what the Metal cgofree group read: whether metal/cmd/serve built with CGO_ENABLED=0, and otool -L's
// output on the binary when otool ran.
type cgoFreeInput struct {
	Built   bool   `json:"built"`
	OtoolOK bool   `json:"otool_ok,omitempty"`
	Otool   string `json:"otool,omitempty"`
}

// gradeMetalCgoFree fails a build that did not succeed, and a binary that links the Metal framework or
// MetalPerformanceShaders at link time: purego-objc resolves them at run time, so a link-time reference falsifies the
// cgo-free claim. When otool did not run there is no linkage to read and the group passes on the build alone.
func gradeMetalCgoFree(in cgoFreeInput) groupVerdict {
	if !in.Built {
		return groupVerdict{State: groupFail, Line: "metal/cmd/serve does not build CGO_ENABLED=0"}
	}
	var lines []string
	if in.OtoolOK {
		for ln := range strings.SplitSeq(in.Otool, "\n") {
			l := strings.ToLower(ln)
			if strings.Contains(l, "metal.framework") || strings.Contains(l, "metalperformanceshaders") {
				lines = append(lines, "        "+strings.TrimSpace(ln))
			}
		}
	}
	if len(lines) > 0 {
		return groupVerdict{State: groupFail, Line: "binary links the Metal framework — the cgo-free claim is false:", Lines: lines}
	}
	return groupVerdict{State: groupPass, Line: "metal/cmd/serve builds CGO_ENABLED=0 and links no Metal framework (dlopen'd via purego-objc)"}
}

// webgpuProbeRecord is the adapter probe's state as a run record holds it (webgpuProbe).
type webgpuProbeRecord struct {
	OptOut  bool   `json:"opt_out,omitempty"`
	Present bool   `json:"present,omitempty"`
	Backend string `json:"backend,omitempty"`
	Failed  string `json:"failed,omitempty"`
}

const webgpuHeader = "W. WebGPU suite + resident parity (adapter-detected, -tags 'gpu goinfer_testhooks')"

// webgpuSuiteCell and webgpuParityCell are the WebGPU group's cells. The parity cell counts top-level results only, and Go
// reports a parent whose subtests all skipped as a top-level PASS, so TestQwen35ResidentParity skips itself when none of
// its subtests ran and the tally sees a real skip. The qwen3.5, granite and nemotron fixtures' weights are gitignored, so
// on a clone without them those tests skip.
var (
	webgpuSuiteCell = cell{Name: "webgpu-suite", Pkgs: []string{"./gpu/", "./gpu/internal/kernels/"}, Tags: []string{"gpu"},
		Serial: true, Extra: []string{"-short"}}
	webgpuParityCell = cell{Name: "webgpu-parity", Pkgs: []string{"./gpu/", "./gpu/internal/kernels/"},
		Tags:    []string{"gpu", "goinfer_testhooks"},
		Run:     webgpuParityRun, // parity.go; checked by TestWebGPUGateIsListedOrExplicitlyNotRequired
		Serial:  true,
		Timeout: "10m",
		Env:     map[string]string{"GOINFER_DNET_PARITY": "1", "GOINFER_SSM_PARITY": "1"}}
)

// gradeWebGPUProbe is the WebGPU group's verdict before any cell runs, and whether the cells run. A probe that never
// answered (it failed to build or run) fails the group: it is not "no adapter".
func gradeWebGPUProbe(p webgpuProbeRecord) (groupVerdict, bool) {
	switch {
	case p.OptOut:
		return groupVerdict{State: groupSkip, Line: "WebGPU group not run: GOINFER_GATE_SKIP_WEBGPU is set"}, false
	case p.Failed != "":
		return groupVerdict{State: groupFail, Line: "WebGPU adapter probe never answered: it failed to build or run, which is not \"no adapter\" (V-21); " +
			"fix the gpu-only test build, or set GOINFER_GATE_SKIP_WEBGPU to opt out"}, false
	case !p.Present:
		return groupVerdict{State: groupSkip, Line: "no WebGPU adapter detected — set GOINFER_GATE_SKIP_WEBGPU to silence this note"}, false
	}
	return groupVerdict{}, true
}

// gradeWebGPUSuite grades the suite cell; the parity cell runs only when it passed.
func gradeWebGPUSuite(backend string, cr cellResult, out string) groupVerdict {
	if cr.RC != 0 {
		return groupVerdict{State: groupFail, Line: fmt.Sprintf("webgpu suite (backend=%s)", backend), Lines: detailLines(out, failLineRe)}
	}
	return groupVerdict{State: groupPass, Line: fmt.Sprintf("webgpu suite (backend=%s)", backend), Lines: evidenceLines(out, okLineRe)}
}

// gradeWebGPUParity grades the parity cell: a non-zero exit, or a cell whose tests all skipped (`go test` exits 0 on one),
// fails it.
func gradeWebGPUParity(cr cellResult, out string) groupVerdict {
	if cr.RC != 0 || (cr.Pass == 0 && cr.Fail == 0 && cr.Skip > 0) {
		return groupVerdict{State: groupFail, Line: "webgpu resident parity gates — a WebGPU forward moved, or nothing ran at all",
			Lines: detailLines(out, failLineRe)}
	}
	return groupVerdict{State: groupPass, Line: "webgpu resident parity gates (qwen3.5 DeltaNet dense; MoE/granite/nemotron opt-in fixtures)",
		Lines: evidenceLines(out, okLineRe)}
}

// gpuExpect is the check groups a run on backend declares, in order.
func gpuExpect(backend string) []string {
	switch backend {
	case "cuda":
		return []string{"cleangpu", "seam", "suite", "parity", "heavy", "graphsforced", "cgofree", "ptx", "webgpu", "repo"}
	case "metal":
		return []string{"cleangpu", "seam", "suite", "parity", "cgofree", "lifecycle", "prefill", "webgpu", "repo"}
	}
	return []string{"cleangpu", "seam", "suite", "webgpu", "repo"}
}

// cleanGPUInput is what the clean-GPU group read: the backend, whether nvidia-smi was found, and its compute processes and
// memory in use ("?" when the query failed).
type cleanGPUInput struct {
	Backend string `json:"backend"`
	HaveSMI bool   `json:"have_smi,omitempty"`
	Procs   string `json:"procs,omitempty"`
	Used    string `json:"used,omitempty"`
}

// gradeCleanGPU skips off CUDA or without nvidia-smi, passes a quiet card or one held only by others (a display server
// legitimately holds some), and fails one a stray goinfer or serve process holds: leaked processes have poisoned a control
// run before, both sides equally, which made a real bug look pre-existing. Its passes do not count toward ran.
func gradeCleanGPU(in cleanGPUInput) groupVerdict {
	if in.Backend != "cuda" || !in.HaveSMI {
		return groupVerdict{State: groupSkip, Line: "clean-GPU check (no nvidia-smi; on Metal check Activity Monitor by hand)"}
	}
	if in.Procs == "" {
		return groupVerdict{State: groupPass, Uncounted: true, Line: fmt.Sprintf("no compute processes on the GPU (%s MiB baseline)", in.Used)}
	}
	pre := []string{"  processes holding the GPU:"}
	for ln := range strings.SplitSeq(in.Procs, "\n") {
		pre = append(pre, "    "+ln)
	}
	low := strings.ToLower(in.Procs)
	if strings.Contains(low, "serve") || strings.Contains(low, "goinfer") || strings.Contains(low, "gi_serve") {
		return groupVerdict{State: groupFail, Pre: pre, Line: "stray goinfer/serve processes hold the GPU — kill them and re-run; leaked processes have\n" +
			"        silently poisoned a control run before (both sides equally, which made a real bug look\n" +
			"        pre-existing). Try: pkill -f '[s]erve'"}
	}
	return groupVerdict{State: groupPass, Uncounted: true, Pre: pre, Line: fmt.Sprintf("no stray goinfer processes (%s MiB in use by others)", in.Used)}
}

// seamCell is the seam group's cell: decoder's serve-decoder-backend seam tests, which need no GPU and no model.
var seamCell = cell{Name: "seam", Pkgs: []string{"./decoder/"}, Run: "TestSeam_"}

// gradeSeam fails a non-zero exit, and a run in which the pattern matched no top-level test: `go test -run NoSuchTest` exits
// 0 and prints "ok", so renaming the tests away would silently delete the check while the gate stayed green.
func gradeSeam(cr cellResult, ran int, out string) groupVerdict {
	if cr.RC != 0 {
		return groupVerdict{State: groupFail, Line: "seam gate — GPU serve may be silently CPU-only (see 7557723 / 727f198)",
			Lines: detailLines(out, failLineRe)}
	}
	if ran == 0 {
		return groupVerdict{State: groupFail, Line: "seam gate: -run 'TestSeam_' matched NO tests — the pattern or the test names moved, and a\n" +
			"        zero-test run exits 0. This gate reported PASS for a check it never executed."}
	}
	return groupVerdict{State: groupPass, Line: "serve↔decoder↔backend seam: residency is actually reached, backend names validate"}
}

// repoInput is what the repo-hygiene group read: the citation lint's exit and output, the CI step list scripts/ci_checks.py
// derived from ci.yml (or why it could not), whether the host is Darwin, and each step it ran, in order.
type repoInput struct {
	LintOK  bool           `json:"lint_ok"`
	LintOut string         `json:"lint_out"`
	RowsOK  bool           `json:"rows_ok"`
	Rows    string         `json:"rows,omitempty"`
	RowsErr string         `json:"rows_err,omitempty"`
	Darwin  bool           `json:"darwin,omitempty"`
	Steps   []ciStepResult `json:"steps,omitempty"`
}

// ciStepResult is one CI step run on this host; Out is kept only for a step that failed, the only output the grader reads.
type ciStepResult struct {
	Job  string `json:"job"`
	Name string `json:"name"`
	OK   bool   `json:"ok"`
	Out  string `json:"out,omitempty"`
}

// ciRow is one row of scripts/ci_checks.py's list: a job, a step name, its kind, an environment override ("-" for none)
// and its command. Runner names a step this host cannot reproduce (kind "runner:<why>").
type ciRow struct {
	Job, Name, Kind, Env, Cmd string
	Runner                    string
}

// ciPlan is the rows this host takes, in order, and how many belong to the other platform. A Darwin host takes the
// *-darwin jobs, any other the linux ones (root, gpu, cuda); the other platform's rows are a counted skip, never dropped.
func ciPlan(rows string, darwin bool) (plan []ciRow, skipped int, other string) {
	mine, other := regexp.MustCompile(`^(root|gpu|cuda)$`), "darwin"
	if darwin {
		mine, other = regexp.MustCompile(`-darwin$`), "linux"
	}
	for line := range strings.SplitSeq(strings.TrimRight(rows, "\n"), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		f := strings.Split(line, "\t")
		if len(f) < 5 {
			continue
		}
		r := ciRow{Job: f[0], Name: f[1], Kind: f[2], Env: f[3], Cmd: f[4]}
		if !mine.MatchString(r.Job) {
			skipped++
			continue
		}
		if after, ok := strings.CutPrefix(r.Kind, "runner:"); ok {
			r.Runner = after
		}
		plan = append(plan, r)
	}
	return plan, skipped, other
}

// gradeRepoLint is the citation lint's verdict: a pass shows the lint's last line, a failure its first six. Its pass does
// not count toward ran.
func gradeRepoLint(ok bool, out string) groupVerdict {
	if ok {
		lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
		return groupVerdict{State: groupPass, Uncounted: true, Line: lines[len(lines)-1]}
	}
	v := groupVerdict{State: groupFail, Line: "docs/QUEUE.md SHA citations"}
	for i, ln := range strings.Split(out, "\n") {
		if i >= 6 {
			break
		}
		v.Lines = append(v.Lines, "      "+ln)
	}
	return v
}

// gradeCIDerivation fails a step list that could not be derived, or came back empty: degrading silently to a hand-written
// list would be the substitution the group exists to prevent, and it would look like a pass.
func gradeCIDerivation(in repoInput) (groupVerdict, bool) {
	if !in.RowsOK || strings.TrimSpace(in.Rows) == "" {
		return groupVerdict{State: groupFail, Line: fmt.Sprintf("cannot derive CI's check set: %s", in.RowsErr)}, false
	}
	return groupVerdict{}, true
}

// gradeCIStep is a planned row's verdict, when it has one: a runner-only row skips naming why, a failed step fails with its
// first five lines, and a passed step has none (it is counted in gradeCISummary).
func gradeCIStep(r ciRow, res ciStepResult) (groupVerdict, bool) {
	if r.Runner != "" {
		return groupVerdict{State: groupSkip, Line: fmt.Sprintf("CI[%s] %s — %s", r.Job, r.Name, r.Runner)}, true
	}
	if res.OK {
		return groupVerdict{}, false
	}
	v := groupVerdict{State: groupFail, Line: fmt.Sprintf("CI[%s] %s", r.Job, r.Name)}
	for i, ln := range strings.Split(res.Out, "\n") {
		if i >= 5 {
			break
		}
		v.Lines = append(v.Lines, "      "+ln)
	}
	return v, true
}

// gradeCISummary is the group's closing verdicts: the other platform's rows as one counted skip, and, when no step failed,
// one pass counting the steps reproduced (not counted toward ran).
func gradeCISummary(skipped int, other string, ok, bad int) []groupVerdict {
	var vs []groupVerdict
	if skipped > 0 {
		vs = append(vs, groupVerdict{State: groupSkip, Line: fmt.Sprintf("%d %s-only CI hygiene step(s) — wrong platform for this host", skipped, other)})
	}
	if bad == 0 {
		vs = append(vs, groupVerdict{State: groupPass, Uncounted: true, Line: fmt.Sprintf("%d CI hygiene check(s) reproduced locally, derived from ci.yml", ok)})
	}
	return vs
}

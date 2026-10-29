package main

import (
	"fmt"
	"strings"
)

// The gpu gate's cgofree (Metal) and webgpu groups as graders (SB12 step 9, docs/tasks/task-subsystems-2026-10.md): each
// verdict is a function of what the run recorded, the build and otool's output for cgofree, the adapter probe and the cells
// for webgpu.

// cgoFreeInput is what the cgofree group read: whether the backend's cmd/serve built with CGO_ENABLED=0, and the linkage
// tool's output on the binary when it ran: otool -L for Metal, ldd and the build's own output for CUDA.
type cgoFreeInput struct {
	Built    bool   `json:"built"`
	OtoolOK  bool   `json:"otool_ok,omitempty"`
	Otool    string `json:"otool,omitempty"`
	BuildOut string `json:"build_out,omitempty"`
	LddOK    bool   `json:"ldd_ok,omitempty"`
	Ldd      string `json:"ldd,omitempty"`
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

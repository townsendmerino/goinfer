package main

import (
	"fmt"
	"strings"
)

// The gpu gate's CUDA cell groups and cgofree as graders (SB12 step 10, docs/tasks/task-subsystems-2026-10.md): suite,
// parity and graphsforced are one `go test` cell each, and each verdict is a function of the cell's graded result, its
// top-level results and text, and what the runner read beside it (the card's free VRAM after a zero-cosine parity failure);
// cgofree's is a function of the build and ldd's recorded output. The live gate and `gate gpu -replay` print through these.

var (
	cudaSuiteCell = cell{Name: "cuda-suite", Pkgs: []string{"./cuda/", "./cuda/kernels/"}, Tags: []string{"cuda"},
		Serial: true, Extra: []string{"-short"}, Env: map[string]string{"CGO_ENABLED": "0"}}
	// The parity cell declares its timeout: without it the cell inherits go's 10m default, and an overrun reports
	// "FAIL <pkg> 608.417s" with no test named, indistinguishable from a crash until detailLines dumps the tail. It is 20m
	// because the package outgrew 10m on a quiet card (docs/measurements/sb12-2026-10.md §19).
	cudaParityCell = cell{Name: "cuda-parity", Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},
		Serial: true, Timeout: "20m", Env: map[string]string{"CGO_ENABLED": "0"}}
	// The graphs cell is separate from parity and labelled FORCED, deliberately. admitGraphs declines under DEFAULT compute
	// mode without MPS, which is correct production behaviour, so on this box capture and replay are otherwise never
	// exercised; forcing them tests the code without changing the admission policy. A forced result must never be read as
	// evidence that graphs are admitted in production, so it stays out of parity.
	cudaGraphsCell = cell{Name: "cuda-graphs", Pkgs: []string{"./cuda/"}, Tags: []string{"cuda", "goinfer_testhooks"},
		Serial: true, Run: "TestGemma4Graphs_",
		Env: map[string]string{"CGO_ENABLED": "0", "GOINFER_CUDA_GRAPHS_UNSAFE": "1"}}
)

const (
	cudaSuiteHeader  = "2a. CUDA kernel-level suite (no testhooks: kernels, admission, lint)"
	cudaParityHeader = "2b. resident PARITY gates (-tags goinfer_testhooks — the forward is asserted here)"
	cudaGraphsHeader = "2d. CUDA graphs bit-exactness, FORCED (GOINFER_CUDA_GRAPHS_UNSAFE=1)"
	cudaCgoHeader    = "3. cgo-free (the whole premise — verify, never assume)"
)

// testNames is the tests' names, in order.
func testNames(ks []testKey) []string {
	out := make([]string, 0, len(ks))
	for _, k := range ks {
		out = append(out, k.Test)
	}
	return out
}

// skipCensusLines lists skipped tests by name, one per line at indent.
func skipCensusLines(skipped []string, indent string) []string {
	lines := make([]string, 0, len(skipped))
	for _, n := range skipped {
		lines = append(lines, indent+"· "+n)
	}
	return lines
}

// gradeCUDASuite grades the kernel-level suite: FAIL with detail on a non-zero exit, PASS with evidence otherwise, then the
// skip census either way. A census of zero is far more likely a broken census than a suite that skipped nothing, so it
// says so.
func gradeCUDASuite(cr cellResult, skipped []string, out string) groupVerdict {
	v := groupVerdict{State: groupPass, Line: "cuda kernel-level suite", Lines: evidenceLines(out, okLineRe)}
	if cr.RC != 0 {
		v = groupVerdict{State: groupFail, Line: "cuda kernel-level suite", Lines: detailLines(out, failLineRe)}
	}
	if len(skipped) == 0 {
		v.Lines = append(v.Lines, "      skip census: 0 — this suite is known to skip several; verify before believing it")
		return v
	}
	v.Lines = append(v.Lines,
		fmt.Sprintf("      skipped within it: %d (all GOINFER_HEAVY_TESTS=1 — bandwidth benchmarks,", len(skipped)),
		"      TestRealWeightGemvParity (real q4_K_M weights), TestResidentSpecServe (loads a 1.5B model))")
	v.Lines = append(v.Lines, skipCensusLines(skipped, "        ")...)
	return v
}

// gradeCUDAParity grades the resident parity cell. A failure whose text holds a cosine of exactly zero also prints the
// all-zero-buffer note with freeVRAM, the card's free memory the runner read after the cell ("" reads "unknown"): an all-zero
// buffer is what a failed allocation leaves, so it is not by itself a parity result. The note states the reading and points
// at the A12 entry; it names no mechanism (docs/code-notes/cmd-gate.md, "vramNote").
func gradeCUDAParity(cr cellResult, out, freeVRAM string) groupVerdict {
	if cr.RC == 0 {
		return groupVerdict{State: groupPass,
			Line:  "resident parity gates (gemma4 dense/two-geom/MoE+router, GLM partial-rotary, mixtral MoE, sliding-window, rope-partial)",
			Lines: evidenceLines(out, okLineRe)}
	}
	v := groupVerdict{State: groupFail, Line: "resident parity gates — a CUDA forward moved. This is the group 2a cannot see.",
		Lines: detailLines(out, failLineRe)}
	if parityZeroCosine(out) {
		if freeVRAM == "" {
			freeVRAM = "unknown"
		}
		v.Lines = append(v.Lines,
			"      NOTE: a cosine of EXACTLY 0.000000 is an all-zero buffer, which is what a failed",
			"            allocation leaves behind — not necessarily a numerics defect.",
			"            free VRAM on the card right now: "+freeVRAM,
			"            (measured AFTER the run, so it is a bound, not the value at failure)",
			"            See docs/QUEUE.md A12. Mechanism NOT established: parallelism (-p 1, one",
			"            package, no t.Parallel) and async teardown (Close is synchronous) are both",
			"            REFUTED, and there is no leak. Do not assume; measure.")
	}
	return v
}

// parityZeroCosine reports whether a failed parity cell's text holds a cosine of exactly zero, the case the runner reads the
// card's free VRAM for.
func parityZeroCosine(out string) bool { return strings.Contains(out, "cosine 0.000000") }

// gradeCUDAGraphs grades the forced-capture graphs cell from its result, how many top-level tests ran, which skipped, and its
// text. A cell that ran no test fails: a zero-test run exits 0. The group always adds its note that capture is FORCED, so a
// pass proves the capture and replay code, not that graphs are admitted in production.
func gradeCUDAGraphs(cr cellResult, ran int, skipped []string, out string) groupVerdict {
	note := []string{"graphs are FORCED in 2d (GOINFER_CUDA_GRAPHS_UNSAFE): this proves the capture/replay CODE, " +
		"not that graphs are admitted in production — admitGraphs still declines here (DEFAULT compute mode, no MPS)."}
	switch {
	case cr.RC != 0:
		return groupVerdict{State: groupFail, Line: "graphs bit-exactness FAILED under forced capture — replay diverges from live launches",
			Lines: detailLines(out, failLineRe), Notes: note}
	case ran == 0:
		return groupVerdict{State: groupFail, Line: "graphs (forced): -run 'TestGemma4Graphs_' matched NO tests — zero-test runs exit 0",
			Notes: note}
	}
	return groupVerdict{State: groupPass,
		Line:  fmt.Sprintf("graphs replay == live launches, FORCED capture (%d tests, %d skipped)", ran, len(skipped)),
		Lines: skipCensusLines(skipped, "        "), Notes: note}
}

// gradeCUDACgoFree fails a build of cuda/cmd/serve that did not succeed, with the build's lines, and a binary whose ldd output
// names a CUDA library: the driver is dlopen'd at run time, so a link-time libcuda, libnvrtc or libcudart falsifies the
// cgo-free claim. A built binary counts toward ran whether or not its linkage then fails. When ldd did not run there is no
// linkage to read and the group passes on the build alone.
func gradeCUDACgoFree(in cgoFreeInput) groupVerdict {
	if !in.Built {
		return groupVerdict{State: groupFail, Line: "cuda/cmd/serve does not build under -tags cuda (CGO_ENABLED=0)",
			Lines: detailLines(in.BuildOut, failLineRe)}
	}
	var lines []string
	if in.LddOK {
		for ln := range strings.SplitSeq(in.Ldd, "\n") {
			l := strings.ToLower(ln)
			if strings.Contains(l, "libcuda") || strings.Contains(l, "libnvrtc") || strings.Contains(l, "libcudart") {
				lines = append(lines, "      "+ln)
			}
		}
	}
	if len(lines) > 0 {
		return groupVerdict{State: groupFail, Line: "binary links CUDA libraries — the cgo-free claim is false:", Lines: lines, Ran: true}
	}
	return groupVerdict{State: groupPass, Line: "serve builds CGO_ENABLED=0 and links no CUDA toolkit (driver is dlopen'd at runtime)"}
}

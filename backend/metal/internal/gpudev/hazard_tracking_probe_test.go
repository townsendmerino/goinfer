//go:build darwin

package gpudev

import (
	"sort"
	"testing"
)

// TestHazardTrackingProbe gates goinfer's audit-metal-2026-09-12.md M-16: every buffer this
// binding allocates is hazard-tracked by default (NewBufferLen and siblings pass options=0), and
// goinfer's own record prices that as PART of a ~3.8 µs/dispatch GPU-side floor across a decode
// token's ~310 dispatches (~1.2 ms of an ~18.5 ms token, ~6%) — but what fraction of that 3.8 µs
// is hazard tracking versus raw per-dispatch overhead was never isolated. This is the
// discriminating probe the finding's own Fix text calls for, in the same spirit as goinfer's own
// M-09/M-10 probes elsewhere in that audit: build the minimal case that varies ONLY the
// buffer-tracking-mode bit, and let the number decide whether the real integration (rewiring a
// resident's actual allocations, in goinfer, across ~40+ call sites) is worth doing at all.
//
// SHAPE: one command buffer, one commit, one wait — decode's own structure, not a Run1D-per-
// dispatch loop, since hazard tracking's cost model is about a SERIAL ENCODER inserting barriers
// between dependent dispatches within one buffer, not about per-submit overhead. 310 dispatches
// cycling through a small fixed set of shared buffers (so the SAME buffer is read/written
// repeatedly across dispatches, the exact pattern that gives Metal's hazard tracker something to
// do — a fresh buffer per dispatch would trivially have no hazard to track either way and prove
// nothing). NOT a claim about production numerics: the kernel is a synthetic elementwise bump, not
// any real decode op — only the dispatch COUNT, buffer COUNT/reuse pattern, and one-command-buffer
// shape are meant to match goinfer's own decode token.
func TestHazardTrackingProbe(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}

	const src = `
#include <metal_stdlib>
using namespace metal;
kernel void bump(device float* buf [[buffer(0)]], constant uint& n [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
    if (tid >= n) return;
    buf[tid] += 1.0f;
}`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "bump")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const nSlots = 20    // distinct buffers reused across dispatches — a decode token's own rough shape
	const nFloats = 1536 // per-buffer size, matching a mid-size hidden-dim scratch buffer
	const nDispatch = 310
	const tg = 256

	q := d.NewCommandQueue()
	runToken := func(bufs []Buffer, nBuf Buffer) float64 {
		e := q.Begin()
		for i := 0; i < nDispatch; i++ {
			e.Dispatch(pipe, nFloats, tg, bufs[i%nSlots], nBuf)
		}
		e.End()
		if err := e.Err(); err != nil {
			t.Fatalf("command buffer aborted: %v", err)
		}
		return e.GPUEnd() - e.GPUStart()
	}

	tracked := make([]Buffer, nSlots)
	untracked := make([]Buffer, nSlots)
	for i := range tracked {
		tracked[i] = d.NewBufferLen(nFloats)
		untracked[i] = d.newBufferLenUntracked("test", nFloats)
	}
	nBuf := NewBufferOf(d, []uint32{uint32(nFloats)})

	const reps = 20
	var trackedMs, untrackedMs []float64
	// Interleaved, not pooled (measurement discipline: a pooled before/after comparison carries
	// thermal/scheduling drift that swamps a small per-dispatch effect).
	for r := 0; r < reps; r++ {
		trackedMs = append(trackedMs, runToken(tracked, nBuf)*1000)
		untrackedMs = append(untrackedMs, runToken(untracked, nBuf)*1000)
	}
	sort.Float64s(trackedMs)
	sort.Float64s(untrackedMs)
	medTracked := trackedMs[reps/2]
	medUntracked := untrackedMs[reps/2]
	perDispatchTracked := medTracked * 1000 / nDispatch
	perDispatchUntracked := medUntracked * 1000 / nDispatch

	t.Logf("tracked:   median %.4f ms/token (%.3f µs/dispatch), min %.4f, max %.4f",
		medTracked, perDispatchTracked, trackedMs[0], trackedMs[reps-1])
	t.Logf("untracked: median %.4f ms/token (%.3f µs/dispatch), min %.4f, max %.4f",
		medUntracked, perDispatchUntracked, untrackedMs[0], untrackedMs[reps-1])
	delta := (medTracked - medUntracked) / medTracked * 100
	t.Logf("untracked vs tracked: %.1f%% of tracked's GPU-busy time (%+.1f%% delta)",
		medUntracked/medTracked*100, delta)
}

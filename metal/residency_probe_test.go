//go:build darwin

package metal

import (
	"testing"
	"time"
)

// TestZZ_residencyProbe isolates the per-submit RESIDENCY cost that dominates the paged-MoE decode
// (~15 ms/boundary of GPU-idle-in-wait, 72× Step-0's 0.213 ms). Three arms separate per-buffer from
// per-byte from re-validation, with trivial GPU work (1 thread) and many buffers REFERENCED (bound +
// read) per command buffer. Load-bearing result (Arm A): a REPEATED identical set caches — submit[0]
// ~70 ms, submit[1..] ~0.4 ms — so the cost is the referenced set CHANGING per submit, and pinning
// the working set resident once (MTLResidencySet / heap useHeap) collapses it. Arms B/C show the
// uncached cost has both a per-buffer term (tracked-allocation count) and a per-byte term. Diagnostic,
// not a gate; documents why the residency-set fix is the lever and would flag an OS residency change.
const residProbeSrc = `
#include <metal_stdlib>
using namespace metal;
kernel void touch16(
  device float* b0 [[buffer(0)]], device float* b1 [[buffer(1)]],
  device float* b2 [[buffer(2)]], device float* b3 [[buffer(3)]],
  device float* b4 [[buffer(4)]], device float* b5 [[buffer(5)]],
  device float* b6 [[buffer(6)]], device float* b7 [[buffer(7)]],
  device float* b8 [[buffer(8)]], device float* b9 [[buffer(9)]],
  device float* b10 [[buffer(10)]], device float* b11 [[buffer(11)]],
  device float* b12 [[buffer(12)]], device float* b13 [[buffer(13)]],
  device float* b14 [[buffer(14)]], device float* b15 [[buffer(15)]],
  uint i [[thread_position_in_grid]]) {
  if (i==0) b0[0] = b0[0]+b1[0]+b2[0]+b3[0]+b4[0]+b5[0]+b6[0]+b7[0]+b8[0]+b9[0]+b10[0]+b11[0]+b12[0]+b13[0]+b14[0]+b15[0];
}`

func TestZZ_residencyProbe(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(residProbeSrc, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "touch16")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	q := d.NewCommandQueue()
	arp := NewARPool()
	defer arp.Drain()

	// one submit that references count buffers (count must be a multiple of 16): ceil(count/16)
	// dispatches, each binding a distinct group of 16 (trivial 1-thread GPU work). Returns wall of
	// waitUntilCompleted and GPU-busy seconds. BeginNP (no per-call pool) so the drain isn't per-submit.
	submit := func(bufs []Buffer) (waitNs, gpuNs int64) {
		e := q.BeginNP()
		for g := 0; g+16 <= len(bufs); g += 16 {
			e.Dispatch(pipe, 1, 1, bufs[g:g+16]...)
		}
		e.FinishEncoding()
		e.Commit()
		t0 := time.Now()
		e.WaitDone()
		waitNs = time.Since(t0).Nanoseconds()
		gpuNs = int64((e.GPUEnd() - e.GPUStart()) * 1e9)
		return
	}
	mkBufs := func(count, bytesEach int) []Buffer {
		b := make([]Buffer, count)
		for i := range b {
			b[i] = d.NewBufferBytes(bytesEach)
		}
		return b
	}
	avgSubmit := func(bufs []Buffer, reps int) (float64, float64) {
		var w, g int64
		for r := 0; r < reps; r++ {
			wn, gn := submit(bufs)
			w += wn
			g += gn
		}
		return float64(w) / 1e6 / float64(reps), float64(g) / 1e6 / float64(reps)
	}

	const MB = 1 << 20

	// ARM A: repeated submits, identical 256×1MB set. Is submit[0] more expensive than submit[1..]?
	a := mkBufs(256, 1*MB)
	var firstW, restW, firstG float64
	for r := 0; r < 12; r++ {
		wn, gn := submit(a)
		w := float64(wn) / 1e6
		if r == 0 {
			firstW, firstG = w, float64(gn)/1e6
		} else {
			restW += w
		}
	}
	restW /= 11
	t.Logf("ARM A (256×1MB, repeated): submit[0] wait %.2f ms (gpu %.2f) | submit[1..11] avg wait %.2f ms — %s",
		firstW, firstG, restW,
		map[bool]string{true: "repeats CHEAPER → residency cached, cost is the set CHANGING", false: "repeats EQUAL → unconditional re-validation"}[restW < firstW*0.7])

	// ARM B: fixed 256 MB total, vary buffer count. 16×16MB vs 256×1MB.
	bFew := mkBufs(16, 16*MB)
	bMany := mkBufs(256, 1*MB)
	wFew, gFew := avgSubmit(bFew, 12)
	wMany, gMany := avgSubmit(bMany, 12)
	t.Logf("ARM B (256 MB total): 16×16MB wait %.2f ms (gpu %.2f, idle %.2f) | 256×1MB wait %.2f ms (gpu %.2f, idle %.2f) — %s",
		wFew, gFew, wFew-gFew, wMany, gMany, wMany-gMany,
		map[bool]string{true: "more buffers COSTLIER at equal bytes → PER-BUFFER (MTLHeap consolidation)", false: "≈equal → not per-buffer"}[wMany-gMany > (wFew-gFew)*1.5])

	// ARM C: fixed 256 buffers, vary bytes. 256×1MB (256 MB) vs 256×4MB (1 GB).
	cSmall := mkBufs(256, 1*MB)
	cBig := mkBufs(256, 4*MB)
	wSmall, gSmall := avgSubmit(cSmall, 12)
	wBig, gBig := avgSubmit(cBig, 12)
	t.Logf("ARM C (256 buffers): 256×1MB wait %.2f ms (idle %.2f) | 256×4MB wait %.2f ms (idle %.2f) — %s",
		wSmall, wSmall-gSmall, wBig, wBig-gBig,
		map[bool]string{true: "more bytes COSTLIER at equal count → PER-BYTE term (residency-once is the only escape)", false: "≈equal → not per-byte"}[wBig-gBig > (wSmall-gSmall)*1.5])
}

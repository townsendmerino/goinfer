package vision

import (
	"runtime"
	"sync"
)

// numCPU is read once; runtime.NumCPU is a syscall-free read but not free.
var numCPU = runtime.NumCPU()

// parallelRows splits [0,rows) into contiguous ranges across cores and runs fn
// on each, when `work` (the total element count) justifies the spawn.
// Otherwise it runs fn(0, rows) inline, so a caller writes one loop body and
// gets both paths. It is the vision twin of encoder's parallelRows.
//
// fn MUST treat its range as exclusively owned. Every caller here writes only
// row-local state, which is what makes the split numerically inert: no
// reduction crosses a row boundary, so each row's f64 mean/variance fold runs
// in exactly the order it did serially.
//
// Why this exists (audit M-10): the only parallelism in this package was the
// P6a head fan-out and the Workspace column split inside projections. Every
// activation, norm, bias and residual pass ran on one goroutine while the
// matmuls around them used every core. On SigLIP-so400m (np=4096, I=4304, 27
// layers) that is 476M GELU-tanh per image — on the order of the tower's entire
// parallel matmul time.
//
// UNLIKE encoder's version there is no in-flight guard here: this package has
// no batch entry point that runs sibling towers concurrently, so there is no
// sibling to yield to. If one is ever added, it needs the same
// inflightForwards check encoder/parallel.go carries, or the two fan-outs will
// oversubscribe.
func parallelRows(rows, work int, fn func(start, end int)) {
	if rows < 2 || work < parallelRowsThreshold {
		fn(0, rows)
		return
	}
	workers := min(numCPU, rows)
	rowsPer := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for w := range workers {
		start := w * rowsPer
		if start >= rows {
			break
		}
		end := min(start+rowsPer, rows)
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			fn(start, end)
		}(start, end)
	}
	wg.Wait()
}

// parallelRowsThreshold is the element count at/above which splitting an
// elementwise pass pays for the goroutine spawn. Same value as encoder's, for
// the same reason: a spawn is ~µs against a pass that is transcendental-bound
// at ~15-20 ns/element.
const parallelRowsThreshold = 1 << 16

// parallelChunks splits a flat elementwise pass (one with no row structure)
// into contiguous chunks and runs fn on each byte range. Elementwise and in
// place, so the split is numerically inert.
func parallelChunks(n int, fn func(lo, hi int)) {
	const chunk = 4096
	chunks := (n + chunk - 1) / chunk
	parallelRows(chunks, n, func(start, end int) {
		fn(start*chunk, min(end*chunk, n))
	})
}

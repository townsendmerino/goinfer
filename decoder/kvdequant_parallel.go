package decoder

import (
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
)

// Audit R-15. An int8-KV decode token widens every stored K and V row of every global layer back to f32 (dequantGlobalLayer), and widens a ring layer's whole
// window the same way. That is O(context) per token per layer, on ONE thread, directly in front of the attention that then fans out across heads: measured on
// Qwen2.5 1.5B at depth 2000, 17.4 ms of an 80.5 ms int8-KV token (the f32 token is 59.5 ms), so the dequantization is essentially the whole int8 slowdown.
//
// The widen is elementwise, float32(q) * scale per row with no shared state, so it splits by rows with no change to any bit. It is bandwidth-bound (1 byte read,
// 4 written per element), so the fan-out is the process's usual width (fusedWorkers: GOMAXPROCS capped by linalg's parallel width) and only kicks in once there is
// enough to amortize a fork/join.

// kvDequantParallel is the A/B handle and test seam for the fan-out, never an environment read: tests and the paired ABBA flip it.
var kvDequantParallel = true

// kvDequantMinFloats is the smallest widen (K and V together) worth a fork/join: 128 Ki floats is 512 KiB of writes, a few tens of microseconds of work.
// A variable only so a test can force the fan-out onto a tiny fixture.
var kvDequantMinFloats = 1 << 17

// kvDequantFanouts counts widens that actually forked: a diagnostic for the tests' non-vacuity check, never branched on.
var kvDequantFanouts atomic.Int64

// dequantKVRows widens rows of K and V (each `cols` wide, one scale per row) into dstK/dstV, split by rows across goroutines when the work is large enough.
// Output is identical to two serial linalg.DequantizeRowsInt8Into calls.
func dequantKVRows(dstK, dstV []float32, kq, vq []int8, ksc, vsc []float32, rows, cols int) {
	if rows <= 0 {
		return
	}
	w := 1
	if kvDequantParallel && 2*rows*cols >= kvDequantMinFloats {
		w = fusedWorkers(rows)
	}
	if w <= 1 {
		linalg.DequantizeRowsInt8Into(dstK, kq, ksc, rows, cols)
		linalg.DequantizeRowsInt8Into(dstV, vq, vsc, rows, cols)
		return
	}
	kvDequantFanouts.Add(1)
	per := (rows + w - 1) / w
	chunk := func(r0, r1 int) {
		o, e := r0*cols, r1*cols
		linalg.DequantizeRowsInt8Into(dstK[o:e], kq[o:e], ksc[r0:r1], r1-r0, cols)
		linalg.DequantizeRowsInt8Into(dstV[o:e], vq[o:e], vsc[r0:r1], r1-r0, cols)
	}
	var wg sync.WaitGroup
	for i := 1; i < w; i++ {
		r0, r1 := i*per, min((i+1)*per, rows)
		if r0 >= r1 {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			chunk(r0, r1)
		}()
	}
	chunk(0, min(per, rows)) // the caller takes the first chunk
	wg.Wait()
}

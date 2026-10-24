package decoder

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
)

// int4FloatActForTest runs every int4 projection with f32 activations instead of the W4A8 kernels' int8 ones: each weight
// row is dequantized (WeightMat.Row) and dotted with the activation as given. It is a measurement seam for the question
// whether int4 weights stay stable once activations are not rounded (docs/tasks/task-multimodal-support-2026-10.md,
// "S19"), and is slow. Set only through SetInt4FloatActForTest (goinfer_testhooks), which also turns the fused W4A8
// batches off, so every int4 projection reaches matmulInt4Float; int4FloatActMissed counts a W4A8 kernel call made while
// it is on, and a gate requires zero.
var int4FloatActForTest bool

var int4FloatActMissed atomic.Int64

// matmulInt4Float is dst[M, rows] = a[M, cols] . w^T with w's int4 rows dequantized to f32 and a unrounded.
func matmulInt4Float(w *linalg.WeightMat, a, dst []float32, M int) {
	rows, cols := w.Rows(), w.Cols()
	workers := min(runtime.GOMAXPROCS(0), rows)
	var wg sync.WaitGroup
	for k := range workers {
		lo, hi := rows*k/workers, rows*(k+1)/workers
		wg.Go(func() {
			buf := make([]float32, cols)
			for r := lo; r < hi; r++ {
				w.Row(r, buf)
				for m := range M {
					dst[m*rows+r] = linalg.Dot(a[m*cols:(m+1)*cols], buf)
				}
			}
		})
	}
	wg.Wait()
}

package cpuengine

import (
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
)

var Int4FloatActMissed atomic.Int64

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

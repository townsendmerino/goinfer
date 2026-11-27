package cpuengine

import (
	"github.com/townsendmerino/aikit/linalg"
)

// inkling2Host is the engine side of an Inkling 2 visit (inkling2.Host): the backend's matmul and the cache's KV rows.
type inkling2Host struct {
	be    Backend
	cache *Cache
}

func (h inkling2Host) MatVec(w *linalg.WeightMat, x, dst []float32) { Matmul(h.be, w, x, dst, 1) }

func (h inkling2Host) AppendKV(l int, k, v []float32) (keys, vals []float32, stride, n int) {
	h.cache.Append(l, k, v)
	keys, vals, stride = h.cache.LayerRows(l, len(k))
	return keys, vals, stride, h.cache.StoredRows(l, len(k))
}

//go:build cuda

package cuda

import (
	"context"
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
)

// kvIdentityIDs is the block table that maps block i to block i, for a contiguous cache of positions rows: KV_ROW(t, k) == k for every
// k < positions. A kernel test that hands a KV kernel a plain [pos][kvDim] buffer passes it as the table.
func kvIdentityIDs(positions int) []int32 {
	ids := make([]int32, max(1, kvBlocksFor(positions)))
	for i := range ids {
		ids[i] = int32(i)
	}
	return ids
}

// kvIdentityTab is kvIdentityIDs on r's device, on the test line. Executor thread (inside r.do) or before any job runs.
func (r *cudaResident) kvIdentityTab(t testing.TB, positions int) Buffer {
	t.Helper()
	ids := kvIdentityIDs(positions)
	b := r.ti(len(ids))
	if e := gpu.Upload(b, ids); e != nil {
		t.Fatalf("upload identity KV table: %v", e)
	}
	return b
}

// gcKVIdentityTab is kvIdentityIDs as a gocudrv buffer, for kernel tests that drive a module directly. The context is synchronised
// after the copy: a pageable copy returns once staged, and a launch on another stream could read the table before it lands.
func gcKVIdentityTab(t testing.TB, cx *gc.Context, positions int) *gc.Buffer[int32] {
	t.Helper()
	ids := kvIdentityIDs(positions)
	b, err := gc.Alloc[int32](cx, len(ids))
	if err != nil {
		t.Fatalf("identity KV table: %v", err)
	}
	if err := b.CopyFrom(context.Background(), ids); err != nil {
		t.Fatalf("identity KV table: %v", err)
	}
	if err := cx.Synchronize(context.Background()); err != nil {
		t.Fatalf("identity KV table: %v", err)
	}
	return b
}

// kvRowsForTest reads positions [lo, hi) of layer l's K (and V, nil for an MLA layer) in slot s through the slot's table, as raw
// float32 words of the cache (an f16 layer's rows come back two halves to a word). Executor thread.
func (r *cudaResident) kvRowsForTest(s, l, lo, hi int) (k, v []float32, err error) {
	words := r.layers[l].kvDim * r.kvElem(l) / 4
	k = make([]float32, (hi-lo)*words)
	if r.vc[l].Len() > 0 {
		v = make([]float32, (hi-lo)*words)
	}
	err = r.kvRowRuns(s, r.kvGroupOf(l), lo, hi, func(p, n, row int) error {
		off, w := (p-lo)*words, n*words
		if e := gpu.Download(r.kc[l].At(row*words*4), k[off:off+w]); e != nil {
			return e
		}
		if v == nil {
			return nil
		}
		return gpu.Download(r.vc[l].At(row*words*4), v[off:off+w])
	})
	return k, v, err
}

// kvWordsToHalves splits kvRowsForTest's words of an f16 layer back into its halves, in memory order.
func kvWordsToHalves(w []float32) []uint16 {
	out := make([]uint16, 2*len(w))
	for i, x := range w {
		b := math.Float32bits(x)
		out[2*i], out[2*i+1] = uint16(b), uint16(b>>16)
	}
	return out
}

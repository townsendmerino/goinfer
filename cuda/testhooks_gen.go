//go:build cuda && goinfer_testhooks

// Code relocated by the B-08 build-tag pass: these are test-only hooks, compiled
// only under -tags goinfer_testhooks so they are NOT part of the public API
// (audit B-08). See RELEASING.md. Mirrors decoder/testhooks_gen.go and gpu/testhooks_gen.go.

package cuda

import gpu "github.com/townsendmerino/aikit/gpu"

// ArgmaxForTest runs the resident's on-device argmax_reduce over caller-supplied logits and returns the
// chosen index — the seam the C-14 tie-break gate uses to feed exact-tie inputs the real forward would
// (almost) never produce. len(logits) must be ≤ r.vocab (it reuses the resident's logits/argIdx
// buffers). Same launch geometry as ForwardArgmax, so it exercises the exact kernel decode runs.
func (r *cudaResident) ArgmaxForTest(logits []float32) (int, error) {
	var id int
	err := r.do(func() error {
		if e := gpu.Upload(r.logits, logits); e != nil {
			return e
		}
		if e := r.launch(r.fArg, onecfg(256, 256*4+256*4), Arg(r.logits),
			gpu.ArgValue(int32(len(logits))), Arg(r.argIdx), Arg(r.argVal)); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		out := make([]int32, 1)
		if e := gpu.Download(r.argIdx, out); e != nil {
			return e
		}
		id = int(out[0])
		return nil
	})
	return id, err
}

// CacheStatsForTest sums the LRU cache hits/misses across all layers (C′ step 2 measurement). A
// miss is one expert's H2D DMA; hit rate = hits/(hits+misses) is the fraction of per-token expert
// bytes reuse saves.
func (r *cudaResident) CacheStatsForTest() (hits, misses uint64) {
	for i := range r.layers {
		if c := r.layers[i].expCache; c != nil {
			hits += c.hits
			misses += c.misses
		}
	}
	return hits, misses
}

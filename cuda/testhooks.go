//go:build cuda && goinfer_testhooks

// Test-only hooks, compiled only under -tags goinfer_testhooks so they are not part of the public API (RELEASING.md).
// Mirrors decoder/testhooks.go and gpu/testhooks.go. Hand-maintained: there is no //go:generate directive and nothing
// regenerates it.

package cuda

import (
	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// ArgmaxForTest runs the resident's on-device argmax_reduce over caller-supplied logits and returns the chosen index,
// the seam the tie-break gate uses to feed exact-tie inputs a real forward would almost never produce. len(logits) must
// be <= r.vocab (it reuses the resident's logits/argIdx buffers); the launch geometry is ForwardArgmax's.
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

// MoeSwigluForTest runs the resident's fused gate||up SwiGLU split (launchGluSplit, the dispatch the routed, shared and
// gemma-4 MoE experts use) over a crafted [gate|up] buffer and returns the f32 pre-quant output (glu_quant's dscratch =
// r.moeScr). It is the wiring gate for a gOff/uOff swap in launchGluSplit, which would compute silu(up)*gate: the
// end-to-end MoE parity cosine cannot catch that on random-weight experts, but it is obvious here where gate != up.
// len(gate) must be <= r.moeInter.
func (r *cudaResident) MoeSwigluForTest(gate, up []float32) ([]float32, error) {
	out := make([]float32, len(gate))
	err := r.do(func() error {
		gu := make([]float32, 2*len(gate))
		copy(gu, gate)
		copy(gu[len(gate):], up)
		if e := gpu.Upload(r.moeGU, gu); e != nil {
			return e
		}
		if e := r.launchGluSplit(r.moeGU, len(gate), r.moeQ, r.moeSc, r.moeScr); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		return gpu.Download(r.moeScr, out)
	})
	return out, err
}

// CacheStatsForTest sums the LRU cache hits/misses across all layers. A miss is one expert's H2D DMA; hits/(hits+misses)
// is the fraction of per-token expert bytes that reuse saves.
func (r *cudaResident) CacheStatsForTest() (hits, misses uint64) {
	for i := range r.layers {
		if c := r.layers[i].expCache; c != nil {
			hits += c.hits
			misses += c.misses
		}
	}
	return hits, misses
}

// PerLayerCacheStatsForTest returns hits/misses for every MoE layer individually, in layer order (dense layers report
// 0/0), the per-layer breakdown CacheStatsForTest's sum discards: is C's per-layer slot budget wasted on some layers and
// starved on others, or is demand even across layers?
func (r *cudaResident) PerLayerCacheStatsForTest() (hits, misses []uint64) {
	hits = make([]uint64, len(r.layers))
	misses = make([]uint64, len(r.layers))
	for i := range r.layers {
		if c := r.layers[i].expCache; c != nil {
			hits[i], misses[i] = c.hits, c.misses
		}
	}
	return hits, misses
}

// PagerStageStatsForTest sums the expert pager's demand accounting across layers: stages is the number of staging events
// (one per routed MoE layer per forward position), distinct the total number of unique experts those stages asked for.
// Zero unless the model runs C' expert staging (GOINFER_MOE_CACHE_EXPERTS). distinct/stages separates two cases that
// wall-clock cannot: if a width-K verify presented all K positions' routing to the pager in one event, the ratio would
// rise with K and a slot budget tuned on decode traffic could be blown by a verify; if verify walks position by
// position, it stays at topK for every K.
func (r *cudaResident) PagerStageStatsForTest() (stages, distinct uint64) {
	for i := range r.layers {
		if c := r.layers[i].expCache; c != nil {
			stages += c.stages
			distinct += c.distinct
		}
	}
	return stages, distinct
}

// ResetPagerStatsForTest zeroes every layer's hit/miss and demand counters so one arm's numbers describe that arm;
// otherwise each arm reports the running total of all before it, a cumulative average that lags hardest where the arms
// differ.
func (r *cudaResident) ResetPagerStatsForTest() {
	for i := range r.layers {
		if c := r.layers[i].expCache; c != nil {
			c.hits, c.misses, c.stages, c.distinct = 0, 0, 0, 0
		}
	}
}

// CacheSlotsForTest is the per-layer slot depth the resident actually built, which is not always the requested one:
// capSlots caps the request to measured free VRAM, so an over-request degrades to fewer slots rather than failing. A
// slot-ladder arm that reported its request would collapse its top rungs into one.
func (r *cudaResident) CacheSlotsForTest() int { return r.cacheSlots }

// LoraCacheStatsForTest reports SetAdapter's bind counters: uploads is how many binds transferred an adapter to the
// device, hits how many reused the cached one.
func (r *cudaResident) LoraCacheStatsForTest() (uploads, hits uint64) {
	_ = r.do(func() error {
		uploads, hits = r.lora.uploads, r.lora.hits
		return nil
	})
	return uploads, hits
}

// TopKForTest runs the resident's on-device topk_select over caller-supplied logits (len ≤ r.vocab) and
// returns the decoded row — the seam the kernel gates use to feed exact-tie, ±0 and short-vocab rows a
// real forward would (almost) never produce. It goes through topkReadback, the same launch-and-unpack
// ForwardTopK runs.
func (r *cudaResident) TopKForTest(logits []float32, k int, temperature float64, wantZ bool) (decoder.TopKRow, error) {
	var row decoder.TopKRow
	err := r.do(func() error {
		if e := gpu.Upload(r.logits, logits); e != nil {
			return e
		}
		var e error
		row, e = r.topkReadback(len(logits), k, temperature, wantZ)
		return e
	})
	return row, err
}

// TopKLaunchForTest runs topk_select over whatever r.logits currently holds (no upload) and decodes the
// row: the per-token cost the decode loop actually pays after a forward, without the H2D the other hook
// adds. Timing instrument only.
func (r *cudaResident) TopKLaunchForTest(v, k int, temperature float64, wantZ bool) error {
	return r.do(func() error {
		_, e := r.topkReadback(v, k, temperature, wantZ)
		return e
	})
}

// GumbelForTest runs the resident's on-device Gumbel-max draw over caller-supplied logits (len ≤ r.vocab) for an
// explicit (seed, draw), returning the id — the seam TestGumbelDeviceAgreesWithHost uses to compare the kernel
// with decoder's reference implementation on rows a real forward would not produce.
func (r *cudaResident) GumbelForTest(logits []float32, temperature float64, seed, draw uint64) (int, error) {
	var id int
	err := r.do(func() error {
		if e := gpu.Upload(r.logits, logits); e != nil {
			return e
		}
		var e error
		id, e = r.gumbelPick(len(logits), float32(1/temperature), seed, draw)
		return e
	})
	return id, err
}

// SetSMShapeForTest makes every cudaResident built after this call size its wave-based fused-projection grids as if the
// device had sms multiprocessors, threadsPerSM resident threads and smemPerSM bytes of shared memory per multiprocessor,
// and returns the restore. It changes only the numbers waveRowsPerWarp sees: the kernels are bit-identical for any
// rows-per-warp, so a decode under a forced shape must equal the real one. Set it BEFORE the model loads; BuildResident
// reads the shape once.
func SetSMShapeForTest(sms, threadsPerSM, smemPerSM int) (restore func()) {
	prev := smShapeOverride
	smShapeOverride = func() (int, int, int, bool) { return sms, threadsPerSM, smemPerSM, true }
	return func() { smShapeOverride = prev }
}

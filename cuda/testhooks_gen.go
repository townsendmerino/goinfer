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

// MoeSwigluForTest runs the resident's fused gate‖up SwiGLU split (launchGluSplit — the SAME dispatch
// the routed / shared / gemma-4 MoE experts use) over a crafted [gate|up] buffer and returns the f32
// pre-quant output (glu_quant's dscratch = r.moeScr). This is the C-15 bug-B wiring gate: a gOff/uOff
// swap in launchGluSplit would compute silu(up)*gate, which the e2e final-logit MoE parity cosine
// CANNOT catch on random-weight experts (silu(up)*gate ≈ silu(gate)*up in magnitude), but is obvious
// here where gate≠up. len(gate) must be ≤ r.moeInter (it reuses the routed-expert scratch buffers).
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

// PagerStageStatsForTest sums the expert pager's DEMAND accounting across layers: stages is the
// number of staging events (one per routed MoE layer per forward POSITION), distinct is the total
// number of unique experts those stages asked for. Zero unless the model runs with C′ expert
// staging (GOINFER_MOE_CACHE_EXPERTS), which is the only configuration that has a pager at all.
//
// distinct/stages is the number the speculation×paging question turns on, and it is worth being
// explicit about what each outcome means, because the two are easy to conflate. If a width-K
// verify presented all K positions' routing to the pager in one event, this ratio would rise with
// K and a slot budget tuned on decode traffic could be blown by a verify — the field-report
// mechanism. If verify instead walks position by position, the ratio stays at topK for every K,
// the slot budget sees exactly the traffic it was tuned on, and any regression that shows up has
// to be explained by something else. The ratio distinguishes them; wall-clock alone cannot.
func (r *cudaResident) PagerStageStatsForTest() (stages, distinct uint64) {
	for i := range r.layers {
		if c := r.layers[i].expCache; c != nil {
			stages += c.stages
			distinct += c.distinct
		}
	}
	return stages, distinct
}

// ResetPagerStatsForTest zeroes every layer's hit/miss and demand counters so one arm's numbers
// describe THAT arm. Without it each arm reports the running total of every arm before it — a
// cumulative average, which lags hardest exactly where an arm differs from its predecessor, i.e.
// at the only place the measurement is trying to look.
func (r *cudaResident) ResetPagerStatsForTest() {
	for i := range r.layers {
		if c := r.layers[i].expCache; c != nil {
			c.hits, c.misses, c.stages, c.distinct = 0, 0, 0, 0
		}
	}
}

// CacheSlotsForTest is the per-layer slot depth the resident actually built, which is NOT always
// the requested one: with GOINFER_MOE_CACHE_SLOTS unset the request is "all experts" and capSlots
// caps it to measured free VRAM, and an over-request degrades to fewer slots rather than failing.
// A slot-ladder arm that reported its REQUEST would silently collapse its top rungs into one.
func (r *cudaResident) CacheSlotsForTest() int { return r.cacheSlots }

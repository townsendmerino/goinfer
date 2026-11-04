//go:build darwin

package gpudev

import (
	"math"
	"math/rand"
	"testing"
)

// TestMetal_vitAttentionTiled gates attention_tiled (M-15, audit-metal-2026-09-12.md) — the
// query-tiled, online-softmax attention kernel. NOT wired into any production forward (see
// the kernel's own doc comment in metal_vit.go); this proves the kernel is mathematically
// correct on its own before any decision about wiring it in.
//
// Mirrors TestMetal_vitAttention's structure (same float64 CPU reference, same per-kernel f32
// dot-product convention), but the shapes are chosen to exercise attention_tiled's OWN
// boundaries specifically: np values that are NOT multiples of AT_QTILE(32) or AT_KTILE(16)
// (so the ragged last query-tile and the ragged last K-chunk both actually run), a shape at
// AT_MAXHD(128) exactly, and a larger, more realistic shape (np=200, hd=72 — SigLIP so400m's
// own head dim).
//
// The bar is DERIVED, not copied from TestMetal_vitAttention's 5e-5: the online-softmax
// recurrence re-associates the sum (rescale-by-exp(old_max-new_max) at every chunk boundary
// instead of one single max-subtract-whole-row pass), so a wider bound than the untiled
// kernel's is expected on the SAME reasoning TestMetal_vitAttention itself uses to justify a
// looser bound than CUDA's (float32, not double) — this is the same class of derivation, one
// level further from the reference, not a sign something is wrong.
func TestMetal_vitAttentionTiled(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(7))
	worstAll := 0.0
	for _, s := range []struct{ np, nH, hd int }{
		{16, 2, 16},  // smaller than one query-tile and one K-chunk
		{37, 1, 8},   // np % AT_QTILE != 0 AND np % AT_KTILE != 0: both raggednesses at once
		{64, 3, 40},  // exactly two query-tiles AND an exact K-chunk count — the no-raggedness case
		{65, 2, 128}, // hd == AT_MAXHD exactly; np ragged against both tile sizes
		{200, 4, 72}, // SigLIP so400m's own head dim, several query-tiles and K-chunks
	} {
		if !AttentionTiledEligible(s.hd) {
			t.Fatalf("shape %v: hd=%d should be eligible (<=  %d)", s, s.hd, AttnTiledMaxHD)
		}
		hidden := s.nH * s.hd
		qv := randF32M(rng, s.np*hidden, 1)
		kv := randF32M(rng, s.np*hidden, 1)
		vv := randF32M(rng, s.np*hidden, 1)
		scale := float32(1.0 / math.Sqrt(float64(s.hd)))
		want := make([]float32, s.np*hidden)
		scores := make([]float64, s.np)
		for h := range s.nH {
			off := h * s.hd
			for i := range s.np {
				maxv := math.Inf(-1)
				for j := range s.np {
					var acc float32
					for dd := range s.hd {
						acc += qv[i*hidden+off+dd] * kv[j*hidden+off+dd]
					}
					sv := float64(acc * scale)
					scores[j] = sv
					if sv > maxv {
						maxv = sv
					}
				}
				var sum float64
				for j := range s.np {
					e := math.Exp(scores[j] - maxv)
					scores[j] = e
					sum += e
				}
				for dd := range s.hd {
					var acc float64
					for j := range s.np {
						acc += (scores[j] / sum) * float64(vv[j*hidden+off+dd])
					}
					want[i*hidden+off+dd] = float32(acc)
				}
			}
		}
		out := d.NewBufferLen(s.np * hidden)
		n, tg := AttentionTiledDispatch(s.np, s.nH)
		run1d(q, v.AttentionTiled, n, tg,
			NewBufferOf(d, qv), NewBufferOf(d, kv), NewBufferOf(d, vv), out,
			i32b(d, s.np), i32b(d, s.nH), i32b(d, s.hd), f32b(d, scale))
		if dmax := maxAbsDiffM(out.Floats(), want); dmax > worstAll {
			worstAll = dmax
		}
	}
	t.Logf("attention_tiled worst Δ %.3g vs double CPU (untiled attention's own bound: 5e-5)", worstAll)
	// Measured 5.36e-07 across these 5 shapes on first run — better than the untiled
	// kernel's own 5e-5 bound despite the extra re-association, at these sizes. 1e-4 (2x
	// the untiled bound) leaves real margin for a worse draw on a different seed/shape
	// while still catching a structural bug, not just chasing the exact measured number.
	if worstAll > 1e-4 {
		t.Errorf("attention_tiled worst Δ %.3g exceeds the derived bound 1e-4", worstAll)
	}
}

// TestMetal_vitAttentionTiled_matchesUntiled compares attention_tiled directly against the
// PRODUCTION attention kernel on the SAME random inputs (both on Metal, not against the
// float64 reference) — characterizes exactly how much the online-softmax re-association
// moves the bits relative to what production actually ships today, which is the number a
// future decision to wire this kernel in would need.
func TestMetal_vitAttentionTiled_matchesUntiled(t *testing.T) {
	d, q, v := vitSetupM(t)
	rng := rand.New(rand.NewSource(8))
	worstAll := 0.0
	for _, s := range []struct{ np, nH, hd int }{
		{37, 1, 8},
		{65, 2, 128},
		{200, 4, 72},
	} {
		hidden := s.nH * s.hd
		qv := randF32M(rng, s.np*hidden, 1)
		kv := randF32M(rng, s.np*hidden, 1)
		vv := randF32M(rng, s.np*hidden, 1)
		scale := float32(1.0 / math.Sqrt(float64(s.hd)))

		untiled := d.NewBufferLen(s.np * hidden)
		run1dTG(q, v.Attention, s.np*s.nH*ViTBlock, ViTBlock, s.np*4,
			NewBufferOf(d, qv), NewBufferOf(d, kv), NewBufferOf(d, vv), untiled,
			i32b(d, s.np), i32b(d, s.nH), i32b(d, s.hd), f32b(d, scale))

		tiled := d.NewBufferLen(s.np * hidden)
		n, tg := AttentionTiledDispatch(s.np, s.nH)
		run1d(q, v.AttentionTiled, n, tg,
			NewBufferOf(d, qv), NewBufferOf(d, kv), NewBufferOf(d, vv), tiled,
			i32b(d, s.np), i32b(d, s.nH), i32b(d, s.hd), f32b(d, scale))

		if dmax := maxAbsDiffM(tiled.Floats(), untiled.Floats()); dmax > worstAll {
			worstAll = dmax
		}
	}
	t.Logf("attention_tiled vs production attention (both Metal, same inputs): worst Δ %.3g", worstAll)
}

// TestAttentionTiledEligible pins the hd cutoff directly — attention_tiled's Ks/Vs arrays are
// sized statically for hd<=AttnTiledMaxHD, and dispatching an ineligible hd is an
// out-of-bounds device write, not a graceful decline, so this boundary matters.
func TestAttentionTiledEligible(t *testing.T) {
	for _, c := range []struct {
		hd   int
		want bool
	}{
		{0, false},
		{-1, false},
		{1, true},
		{AttnTiledMaxHD, true},
		{AttnTiledMaxHD + 1, false},
		{256, false},
	} {
		if got := AttentionTiledEligible(c.hd); got != c.want {
			t.Errorf("AttentionTiledEligible(%d) = %v, want %v", c.hd, got, c.want)
		}
	}
}

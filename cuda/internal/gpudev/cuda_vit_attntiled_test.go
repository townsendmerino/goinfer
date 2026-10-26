//go:build linux

package gpudev

import (
	"math"
	"math/rand"
	"testing"
)

// TestCUDA_vitAttentionTiled gates attention_tiled on CUDA — the query-tiled,
// online-softmax (FlashAttention-style) attention kernel.
//
// Exercises small shapes, ragged shapes (np % AT_QTILE != 0 and np % AT_KTILE != 0),
// AT_MAXHD (128), and SigLIP-so400m's hd (72).
func TestCUDA_vitAttentionTiled(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(7))
	worstAll := 0.0
	for _, s := range []struct{ np, nH, hd int }{
		{16, 2, 16},  // smaller than one query-tile and one K-chunk
		{37, 1, 8},   // np % AT_QTILE != 0 AND np % AT_KTILE != 0: both raggednesses at once
		{64, 3, 40},  // exactly two query-tiles AND an exact K-chunk count
		{65, 2, 128}, // hd == AT_MAXHD exactly; np ragged against both tile sizes
		{200, 4, 72}, // SigLIP so400m's head dim, several query-tiles and K-chunks
	} {
		if !AttentionTiledEligible(s.hd) {
			t.Fatalf("shape %v: hd=%d should be eligible (<= %d)", s, s.hd, AttnTiledMaxHD)
		}
		hidden := s.nH * s.hd
		qv := randF32(rng, s.np*hidden, 1)
		kv := randF32(rng, s.np*hidden, 1)
		vv := randF32(rng, s.np*hidden, 1)
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
		cfg := AttentionTiledLaunchConfig(s.np, s.nH)
		if err := q.Launch(v.AttentionTiled, cfg,
			Arg(NewBufferOf(d, qv)), Arg(NewBufferOf(d, kv)), Arg(NewBufferOf(d, vv)), Arg(out),
			ArgValue(int32(s.np)), ArgValue(int32(s.nH)), ArgValue(int32(s.hd)), ArgValue(scale)); err != nil {
			t.Fatalf("shape %v: Launch failed: %v", s, err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("shape %v: Sync failed: %v", s, err)
		}
		got := make([]float32, s.np*hidden)
		if err := Download(out, got); err != nil {
			t.Fatalf("shape %v: Download failed: %v", s, err)
		}
		if dmax := maxAbsDiff(got, want); dmax > worstAll {
			worstAll = dmax
		}
	}
	t.Logf("attention_tiled worst Δ %.3g vs double CPU (bound 1e-4)", worstAll)
	if worstAll > 1e-4 {
		t.Errorf("attention_tiled worst Δ %.3g exceeds bound 1e-4", worstAll)
	}
}

// TestCUDA_vitAttentionTiled_matchesUntiled compares attention_tiled directly against the
// production attention kernel on the SAME random inputs.
func TestCUDA_vitAttentionTiled_matchesUntiled(t *testing.T) {
	d, q, v := vitSetup(t)
	rng := rand.New(rand.NewSource(99))
	worstAll := 0.0
	for _, s := range []struct{ np, nH, hd int }{
		{16, 2, 16},
		{37, 1, 8},
		{64, 3, 40},
		{65, 2, 128},
		{200, 4, 72},
	} {
		hidden := s.nH * s.hd
		qv := randF32(rng, s.np*hidden, 1)
		kv := randF32(rng, s.np*hidden, 1)
		vv := randF32(rng, s.np*hidden, 1)
		scale := float32(1.0 / math.Sqrt(float64(s.hd)))

		outUntiled := d.NewBufferLen(s.np * hidden)
		if err := q.Launch(v.Attention, AttentionGrid(s.np, s.nH),
			Arg(NewBufferOf(d, qv)), Arg(NewBufferOf(d, kv)), Arg(NewBufferOf(d, vv)), Arg(outUntiled),
			ArgValue(int32(s.np)), ArgValue(int32(s.nH)), ArgValue(int32(s.hd)), ArgValue(scale)); err != nil {
			t.Fatalf("shape %v untiled Launch: %v", s, err)
		}

		outTiled := d.NewBufferLen(s.np * hidden)
		if err := q.Launch(v.AttentionTiled, AttentionTiledLaunchConfig(s.np, s.nH),
			Arg(NewBufferOf(d, qv)), Arg(NewBufferOf(d, kv)), Arg(NewBufferOf(d, vv)), Arg(outTiled),
			ArgValue(int32(s.np)), ArgValue(int32(s.nH)), ArgValue(int32(s.hd)), ArgValue(scale)); err != nil {
			t.Fatalf("shape %v tiled Launch: %v", s, err)
		}
		if err := q.Sync(); err != nil {
			t.Fatalf("shape %v Sync: %v", s, err)
		}

		gotUntiled := make([]float32, s.np*hidden)
		gotTiled := make([]float32, s.np*hidden)
		if err := Download(outUntiled, gotUntiled); err != nil {
			t.Fatal(err)
		}
		if err := Download(outTiled, gotTiled); err != nil {
			t.Fatal(err)
		}

		if dmax := maxAbsDiff(gotTiled, gotUntiled); dmax > worstAll {
			worstAll = dmax
		}
	}
	t.Logf("attention_tiled vs untiled attention: worst Δ %.3g", worstAll)
	if worstAll > 1e-4 {
		t.Errorf("worst Δ %.3g exceeds 1e-4", worstAll)
	}
}

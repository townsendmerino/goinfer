//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// TestGeomVariants_dedup is the P1 (own-forward residency bridge) unit gate for the
// value-keyed attention-geometry dedup. TestDecodeToken_parity already proves the
// *collapse* direction (a uniform multi-layer model dedups to one attnGeom, so
// non-Gemma models stay byte-identical). This proves the *expand* direction and that
// the key actually depends on every field: two layers whose {hd, nKV, half} tuples
// differ in ANY one component must produce two distinct geoms, not one. If the key
// silently dropped a field, the matching case below would collapse to a single variant.
//
// It asserts GeomVariantCount only — the plan is built, never run — so the second
// layer's tuple may diverge from its (uniform) weight shapes without consequence; the
// dedup count is decided at build time from the tuples alone.
func TestGeomVariants_dedup(t *testing.T) {
	ctx := newOrSkipHW(t)
	defer ctx.Close()

	const hidden, nH, nKV, hd, inter, vocab, pos = 256, 4, 2, 64, 512, 128, 3
	qDim, kvDim := nH*hd, nKV*hd
	half := hd / 2
	eps := float32(1e-6)
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	// A fresh 2-layer uniform W8A8 model each call (fresh device buffers so each runner
	// owns its own inputs). The geometry-bearing weights are uniform; only the runLayer
	// tuple we inject below distinguishes the layers.
	seed := uint64(1)
	W := func(N, K int) *ResidentW8A8 {
		seed++
		bq, s := quantW(N, K, seed)
		rm, e := ctx.UploadW8A8(bq, s, N, K)
		if e != nil {
			t.Fatal(e)
		}
		return rm
	}
	up32 := func(v []float32) *DeviceBuffer {
		d, e := ctx.UploadF32(v)
		if e != nil {
			t.Fatal(e)
		}
		return d
	}
	invFreq := make([]float32, half)
	for d := range invFreq {
		invFreq[d] = float32(1.0 / math.Pow(1e6, float64(2*d)/float64(hd)))
	}
	buildMW := func() ModelW {
		invD := up32(invFreq)
		mw := ModelW{FinalNorm: up32(randMat(hidden, 600)), LMHead: W(vocab, hidden)}
		defer mw.Release() // resident weights are caller-owned; Context.Close does not free them
		for l := range 2 {
			kc, _ := ctx.NewKVCache(nil, (pos+1)*kvDim)
			vc, _ := ctx.NewKVCache(nil, (pos+1)*kvDim)
			mw.Layers = append(mw.Layers, LayerW{
				Attn: AttnWeights{
					Norm: up32(randMat(hidden, uint64(200+l))), QProj: W(qDim, hidden), KProj: W(kvDim, hidden),
					VProj: W(kvDim, hidden), OProj: W(hidden, qDim), InvFreq: invD, KCache: kc, VCache: vc,
				},
				MLPNorm: up32(randMat(hidden, uint64(300+l))), Gate: W(inter, hidden), Up: W(inter, hidden), Down: W(hidden, inter),
			})
		}
		return mw
	}

	// count builds a runner from a fresh model with layer 1's geometry set to
	// (ghd, gnKV, ghalf) + kEqV and reports how many distinct geometries the plan dedups
	// to. ghd==0 leaves the dims at the model default (used to vary kEqV in isolation).
	count := func(t *testing.T, ghd, gnKV, ghalf int, kEqV bool) int {
		rm := w8Model(buildMW())
		if ghd != 0 {
			rm.layers[1].ghd, rm.layers[1].gnKV, rm.layers[1].ghalf = ghd, gnKV, ghalf
		}
		rm.layers[1].gKEqV = kEqV
		runner, err := ctx.newDecodeRunner(rm, hidden, nH, nKV, hd, inter, 0, eps, scale, false)
		if err != nil {
			t.Fatalf("newDecodeRunner: %v", err)
		}
		defer runner.Release()
		return runner.GeomVariantCount()
	}

	// Baseline: both layers fall back to the model geometry ⇒ one variant.
	if gv := count(t, 0, 0, 0, false); gv != 1 {
		t.Errorf("uniform 2-layer: GeomVariantCount = %d, want 1", gv)
	}

	// Differ in exactly one field ⇒ two variants. Each case would collapse to 1 if the
	// key omitted that field. Layer 0 keeps the model tuple (hd=64, nKV=2, half=32,
	// kEqV=false). The kEqV case keeps the dims at default and flips only the flag, so it
	// fails if attention_k_eq_v is absent from the key — the soundness bug it guards.
	for _, tc := range []struct {
		name             string
		ghd, gnKV, ghalf int
		kEqV             bool
	}{
		{"hd", 128, nKV, half, false}, // wider head (global-like)
		{"nKV", hd, 1, half, false},   // fewer KV heads
		{"half", hd, nKV, 16, false},  // narrower rotary
		{"kEqV", 0, 0, 0, true},       // same dims, K=V flag only
	} {
		t.Run(tc.name, func(t *testing.T) {
			if gv := count(t, tc.ghd, tc.gnKV, tc.ghalf, tc.kEqV); gv != 2 {
				t.Errorf("layer 1 differs in %s: GeomVariantCount = %d, want 2", tc.name, gv)
			}
		})
	}
}

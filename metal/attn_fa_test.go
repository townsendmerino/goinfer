//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestAttentionFA_vsReference is R2's (docs/tasks/red-october.md) gate (1), basic form: the new
// attention_fa/attention_fa_combine pair against the CPU f64-shaped reference (cpuAttention,
// shared with TestAttention_ShippedKernelShapes), at the "control qwen2.5-1.5b" shape (hd=128,
// the only head width this kernel supports — see its own doc comment) and a handful of nKeys/nSplit
// combinations. Not yet the amended runAttnCase-pattern cases (hot key at split boundaries, a
// rising-score ramp) — this is the first correctness pass, proving the mechanism before the harder
// adversarial inputs.
func TestAttentionFA_vsReference(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pFA, err := d.NewComputePipeline(lib, "attention_fa")
	if err != nil {
		t.Fatalf("pipeline attention_fa: %v", err)
	}
	pCombine, err := d.NewComputePipeline(lib, "attention_fa_combine")
	if err != nil {
		t.Fatalf("pipeline attention_fa_combine: %v", err)
	}

	const nH, nKV, hd = 12, 2, 128 // qwen2.5-1.5b shape — R2's registered decision cell
	G := nH / nKV

	for _, tc := range []struct {
		what          string
		nKeys, nSplit int
		hotKey        int // -1 = default (nKeys/2)
		ramp          bool
		window        uint32
	}{
		{"K=24 S=1", 24, 1, -1, false, 0},
		{"K=24 S=2", 24, 2, -1, false, 0},
		{"K=128 S=1", 128, 1, -1, false, 0},
		{"K=128 S=4", 128, 4, -1, false, 0},
		{"K=2048 S=1", 2048, 1, -1, false, 0},
		{"K=2048 S=8", 2048, 8, -1, false, 0},
		{"K=3900 S=14", 3900, 14, -1, false, 0}, // kvHead(2) x S(14) = 28 >= 2x core count, per the brief's own rule
		{"K=600 windowed S=4", 600, 4, -1, false, 512},
		// R2's amended gate (1): the hot key at the FIRST split, the LAST split, and each side of
		// a split boundary -- a defect in the cross-split combine or the cross-simdgroup combine
		// would show up as a wrong-magnitude output exactly when the dominant key sits at one of
		// these edges, and nowhere else (the general random-hot-key-at-middle cases above cannot
		// see it, same lesson TestAttention_ShippedKernelShapes' own header states).
		{"K=2048 S=8 hotKey=first split", 2048, 8, 0, false, 0},
		{"K=2048 S=8 hotKey=last split", 2048, 8, 2047, false, 0},
		{"K=2048 S=8 hotKey=split boundary-1", 2048, 8, 255, false, 0}, // last key of split 0 (chunkLen=256)
		{"K=2048 S=8 hotKey=split boundary", 2048, 8, 256, false, 0},   // first key of split 1
		{"K=2048 S=4 hotKey=split boundary-1", 2048, 4, 511, false, 0}, // chunkLen=512
		{"K=2048 S=4 hotKey=split boundary", 2048, 4, 512, false, 0},
		// A rising score ramp: the running max moves at EVERY step within a simdgroup's own walk,
		// not just once -- exercises the alpha-rescale repeatedly, at each S in {1,2,4}.
		{"K=2048 S=1 rising ramp", 2048, 1, -1, true, 0},
		{"K=2048 S=2 rising ramp", 2048, 2, -1, true, 0},
		{"K=2048 S=4 rising ramp", 2048, 4, -1, true, 0},
	} {
		t.Run(tc.what, func(t *testing.T) {
			cos, maxabs := runAttnFACase(t, d, pFA, pCombine, nH, nKV, hd, G, tc.nKeys, tc.nSplit, tc.hotKey, tc.ramp, tc.window)
			t.Logf("%s: cosine=%.7f maxAbs=%.2e", tc.what, cos, maxabs)
			mustFinite(t, tc.what+" cosine", cos)
			if cos < 0.9999 || maxabs > 1e-2 {
				t.Errorf("%s: attention_fa parity FAIL cosine=%.7f maxAbs=%.2e", tc.what, cos, maxabs)
			}
		})
	}
}

// runAttnFACase mirrors runAttnCase's realistic-pattern construction (attn_shape_test.go) --
// sharp hot key, near-zero sink, outlier V row -- against the SAME cpuAttention reference, but
// dispatches attention_fa + attention_fa_combine instead of the shipped attention kernel. hotKey
// >= 0 overrides the default nKeys/2 placement (R2's amended gate (1): the hot key must be driven
// at the first split, the last split, and each side of a split boundary -- a diffuse/random-only
// input cannot see a rescale or combine defect, exactly the lesson attn_shape_test.go's own header
// already states for the shipped kernel).
func runAttnFACase(t *testing.T, d *Device, pFA, pCombine Pipeline, nH, nKV, hd, G, nKeys, nSplit, hotKey int, ramp bool, window uint32) (cos, maxabs float64) {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(hd*1000 + nKeys*7 + nSplit)))
	kvDim := nKV * hd
	scale := float32(1 / math.Sqrt(float64(hd)))

	q := make([]float32, nH*hd)
	for i := range q {
		q[i] = float32(rng.NormFloat64()) * 0.5
	}
	kf := make([]float32, nKeys*kvDim)
	vf := make([]float32, nKeys*kvDim)
	for i := range kf {
		kf[i] = float32(rng.NormFloat64()) * 0.5
		vf[i] = float32(rng.NormFloat64()) * 0.5
	}
	for i := range kvDim {
		vf[i] *= 0.05 // sink-like near-zero V at key 0
	}
	if ramp {
		// A RISING score ramp: each key's K scaled up with position, so the running max moves at
		// every step (not just once) -- the case that actually exercises the online-softmax
		// rescale (alpha) repeatedly within a simdgroup's own key walk, not just at combine time.
		for s := range nKeys {
			g := 1.0 + float32(s)/float32(nKeys)
			for h := range nKV {
				for dd := range hd {
					kf[s*kvDim+h*hd+dd] *= g
				}
			}
		}
	}
	hot := nKeys / 2
	if hotKey >= 0 {
		hot = hotKey
	}
	for h := range nKV {
		for dd := range hd {
			kf[hot*kvDim+h*hd+dd] = q[(h*(nH/nKV))*hd+dd] * 3
		}
	}
	for i := range kvDim {
		vf[(nKeys-2)*kvDim+i] *= 20 // outlier V row
	}

	kh, vh := make([]uint16, len(kf)), make([]uint16, len(vf))
	for i := range kf {
		kh[i], vh[i] = f32ToF16(kf[i]), f32ToF16(vf[i])
	}
	for i := range kf {
		kf[i], vf[i] = f16ToF32(kh[i]), f16ToF32(vh[i])
	}

	ref := cpuAttention(q, kf, vf, nH, nKV, hd, nKeys, int(window), scale)

	qB := NewBufferFloats(d, q)
	kc, vc := NewBufferU16s(d, kh), NewBufferU16s(d, vh)
	pStride := G * (hd + 2)
	partial := d.NewBufferLen(nKV * nSplit * pStride)
	out := d.NewBufferLen(nH * hd)
	uNKV, uG := NewBufferU32(d, uint32(nKV)), NewBufferU32(d, uint32(G))
	uNKeys := NewBufferU32(d, uint32(nKeys))
	uScale := NewBufferFloats(d, []float32{scale})
	uWin := NewBufferU32(d, window)
	uNSplit := NewBufferU32(d, uint32(nSplit))
	uHd := NewBufferU32(d, uint32(hd))

	shmBytes := 128 * 6 * G * 4 // 128 threads * (m[G]+l[G]+acc[G][4]) floats * 4 bytes

	cq := d.NewCommandQueue()
	enc := cq.Begin()
	enc.DispatchTG(pFA, nKV*nSplit*128, 128, shmBytes, qB, kc, vc, partial, uNKV, uG, uNKeys, uScale, uWin, uNSplit)
	enc.Dispatch(pCombine, nH*hd, hd, partial, out, uG, uHd, uNSplit)
	enc.End()

	got := out.Floats()
	var dot, na, nb float64
	for i := range ref {
		dot += float64(got[i]) * float64(ref[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(ref[i]) * float64(ref[i])
		if dd := math.Abs(float64(got[i] - ref[i])); dd > maxabs {
			maxabs = dd
		}
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30), maxabs
}

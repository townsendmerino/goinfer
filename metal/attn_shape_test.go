//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestAttention_ShippedKernelShapes drives the SHIPPED attention kernel (allKernels, the one
// model.go dispatches) at Gemma 3's exact attention shape and at the control's, on REALISTIC
// attention patterns — the case the existing coverage structurally cannot reach.
//
// Why this gap exists. Metal's gemma3 parity carries a Gemma-only residual of -0.104 against its
// own CPU-int4 twin (metal/quantbar_test.go) that the weights do not explain (the double
// quantization measured free — decoder/requant_test.go), so a Gemma-specific kernel is wrong. The
// shape of the residual points here: 9 gaps >3% with a worst near-tie of 40.8% is not what a
// per-layer precision delta compounding over 34 layers looks like (that is a smooth droop) — it
// is an op failing on PARTICULAR positions. And multi-key attention at hd=256 is the one op that
// is both Gemma-specific and untested:
//
//   - the dense control is hd=128, so it is blind to an hd=256 fault by construction;
//   - attention_test.go compiles its own INLINE copy of the kernel, not the shipped source;
//   - layer_test.go runs the shipped one, but at hd=64 — and does not bind `window` at all;
//   - the position-0 analysis could never see it: at pos 0 attention output IS v0 exactly, so
//     multi-key softmax never runs. Every Gemma number that mattered lives at pos>0.
//
// The patterns matter as much as the shape. Random q/k give a DIFFUSE softmax where every key
// contributes ~1/nKeys and errors average out — which is why a random-weight synthetic passed at
// 0.997 while the real model does not. Real attention is SHARP (one key dominant) and real V
// carries outliers, so this drives both: sharp scores, an outlier V row, and Gemma's own
// near-zero-norm sink at key 0.
//
// The reference rounds K/V to f16 first, because the kernel reads an f16 KV cache — comparing
// against an f32 reference would measure the cache's precision, not the kernel's correctness.
func TestAttention_ShippedKernelShapes(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pAttn, err := d.NewComputePipeline(lib, "attention")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	for _, tc := range []struct {
		what               string
		nH, nKV, hd, nKeys int
		window             uint32
	}{
		{"control qwen2.5-1.5b (hd=128)", 12, 2, 128, 24, 0},
		{"gemma3-4b global (hd=256)", 8, 4, 256, 24, 0},
		{"gemma3-4b local (hd=256, win=1024)", 8, 4, 256, 24, 1024},
		{"gemma3-4b global, long ctx", 8, 4, 256, 600, 0},
		{"gemma3-4b local, window ENGAGED", 8, 4, 256, 600, 512},
	} {
		t.Run(tc.what, func(t *testing.T) {
			cos, maxabs, gn, rn := runAttnCase(t, d, pAttn, tc.nH, tc.nKV, tc.hd, tc.nKeys, tc.window)
			// Norm is reported next to cosine deliberately: the sink hunt burned a week on a
			// cosine that was meaningless because the vector under it was near-zero. Never again
			// read one without the other.
			t.Logf("%s: cosine=%.7f maxAbs=%.2e |gpu|=%.4f |cpu|=%.4f", tc.what, cos, maxabs, gn, rn)
			mustFinite(t, tc.what+" cosine", cos)
			if cos < 0.9999 || maxabs > 1e-2 {
				t.Errorf("%s: attention parity FAIL cosine=%.7f maxAbs=%.2e", tc.what, cos, maxabs)
			}
		})
	}
}

// runAttnCase builds one realistic attention step, runs the shipped kernel, and scores it
// against a CPU reference computed over the SAME f16-rounded K/V.
func runAttnCase(t *testing.T, d *Device, pAttn Pipeline, nH, nKV, hd, nKeys int, window uint32) (cos, maxabs, gn, rn float64) {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(hd*1000 + nKeys)))
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
	// Gemma's <bos> is an attention SINK: its V is trained near-zero (|V| 9.4 vs 129 typical).
	for i := 0; i < kvDim; i++ {
		vf[i] *= 0.05
	}
	// A SHARP softmax: make one mid-sequence key align strongly with q, so the output is
	// dominated by a single V row rather than an average of all of them.
	hot := nKeys / 2
	for h := 0; h < nKV; h++ {
		for dd := 0; dd < hd; dd++ {
			kf[hot*kvDim+h*hd+dd] = q[(h*(nH/nKV))*hd+dd] * 3
		}
	}
	// An OUTLIER V row — real value vectors are not homogeneous.
	for i := 0; i < kvDim; i++ {
		vf[(nKeys-2)*kvDim+i] *= 20
	}

	// Round K/V to f16 (what the cache holds), then reference off the ROUNDED values.
	kh, vh := make([]uint16, len(kf)), make([]uint16, len(vf))
	for i := range kf {
		kh[i], vh[i] = f32ToF16(kf[i]), f32ToF16(vf[i])
	}
	for i := range kf {
		kf[i], vf[i] = f16ToF32(kh[i]), f16ToF32(vh[i])
	}

	ref := cpuAttention(q, kf, vf, nH, nKV, hd, nKeys, int(window), scale)

	qB := d.NewBufferFloats(q)
	kc, vc := d.NewBufferU16s(kh), d.NewBufferU16s(vh)
	out := d.NewBufferLen(nH * hd)
	uNH, uNKV := d.NewBufferU32(uint32(nH)), d.NewBufferU32(uint32(nKV))
	uHd, uNKeys := d.NewBufferU32(uint32(hd)), d.NewBufferU32(uint32(nKeys))
	uScale := d.NewBufferFloats([]float32{scale})
	uWin := d.NewBufferU32(window)

	cq := d.NewCommandQueue()
	enc := cq.Begin()
	enc.Dispatch(pAttn, nH*128, 128, qB, kc, vc, out, uNH, uNKV, uHd, uNKeys, uScale, uWin)
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
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30), maxabs, math.Sqrt(na), math.Sqrt(nb)
}

// cpuAttention is the plain-Go reference: GQA, causal, optional sliding window, f32 accumulation.
func cpuAttention(q, kf, vf []float32, nH, nKV, hd, nKeys, window int, scale float32) []float32 {
	kvDim := nKV * hd
	out := make([]float32, nH*hd)
	winStart := 0
	if window > 0 && nKeys > window {
		winStart = nKeys - window
	}
	for qh := 0; qh < nH; qh++ {
		kvh := qh / (nH / nKV)
		sc := make([]float32, nKeys)
		mx := float32(math.Inf(-1))
		for s := winStart; s < nKeys; s++ {
			var a float32
			for dd := 0; dd < hd; dd++ {
				a += q[qh*hd+dd] * kf[s*kvDim+kvh*hd+dd]
			}
			sc[s] = a * scale
			if sc[s] > mx {
				mx = sc[s]
			}
		}
		var sum float32
		for s := winStart; s < nKeys; s++ {
			sc[s] = float32(math.Exp(float64(sc[s] - mx)))
			sum += sc[s]
		}
		for dd := 0; dd < hd; dd++ {
			var a float32
			for s := winStart; s < nKeys; s++ {
				a += sc[s] * vf[s*kvDim+kvh*hd+dd]
			}
			out[qh*hd+dd] = a / sum
		}
	}
	return out
}

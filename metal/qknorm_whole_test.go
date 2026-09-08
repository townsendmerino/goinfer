//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestQKNorm_wholeVector is G5's real gate for FeatQKNormWhole (docs/task-gpu-paths-2026-09.md,
// Olmo 3/Olmo Hybrid): unlike the last two G5 rows, this feature has no new pure-Go formula to
// unit-test — the whole change is a DISPATCH-GEOMETRY reinterpretation of the ALREADY-SHIPPED
// per-head qk_norm kernel (encodeAttention passes nH=1,nKV=1,hd=nH_orig*hd_orig instead of the
// per-head geometry). This test proves that reinterpretation directly against the real kernel
// (compiled from allKernels, the production MSL source — same discipline as TestQKNorm above),
// with an EXACT per-component comparison against a whole-vector CPU RMSNorm reference (rows=1,
// dim=nH*hd, mirroring decoder/attention.go's rmsNorm(q,QNorm,1,nH*hd,...) exactly) — no GPU
// quantization noise anywhere in this path (the kernel takes and returns plain f32), so this is
// as decisive as G5 rows 1-2's pure decoder-level unit tests, just one level lower (kernel
// instead of Go function).
func TestQKNorm_wholeVector(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Fatalf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "qk_norm")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	const nH, hd = 8, 8 // Olmo3-tiny's real shape: 8 heads, head_dim 8 (hidden=64)
	const eps = 1e-6
	wide := nH * hd // the WHOLE vector width qk_norm now reduces over in one block
	nHhd := nH * hd
	kvDim := nH * hd // MHA: nKV == nH, same width
	qkvDim := nHhd + 2*kvDim
	rng := rand.New(rand.NewSource(17))
	rvec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}

	for _, addOne := range []bool{false, true} {
		qkv := rvec(qkvDim)
		qn, kn := rvec(wide), rvec(wide) // whole-vector weight: nH*hd wide, not per-head hd
		// CPU reference: ONE RMS statistic over the WHOLE nH*hd-wide vector, applied per-element —
		// decoder/attention.go's rmsNorm(q, QNorm, rows=1, dim=nH*hd, ...), not the per-head form.
		ref := append([]float32(nil), qkv...)
		normWhole := func(off int, w []float32) {
			var ss float64
			for i := range wide {
				ss += float64(ref[off+i]) * float64(ref[off+i])
			}
			r := 1 / math.Sqrt(ss/float64(wide)+eps)
			for i := range wide {
				wt := w[i]
				if addOne {
					wt = 1 + w[i]
				}
				ref[off+i] = float32(float64(ref[off+i]) * r * float64(wt))
			}
		}
		normWhole(0, qn)    // whole Q vector
		normWhole(nHhd, kn) // whole K vector (V section, off nHhd+kvDim, untouched)
		ao := uint32(0)
		if addOne {
			ao = 1
		}
		q := d.NewCommandQueue()
		buf := NewBufferFloats(d, qkv)
		// nH=1, nKV=1, hd=wide: block h=0 reduces [0,wide) (all of Q), block h=1 reduces
		// [nHhd,nHhd+wide) (all of K) — exactly the geometry metal/model.go's encodeAttention
		// dispatches when r.qkNormWhole. Grid is (1+1)*128, matching (qkNH+qkNKV)*tgReduceAttn.
		q.Run1D(pipe, 2*128, 128, buf, NewBufferFloats(d, qn), NewBufferFloats(d, kn),
			NewBufferU32(d, 1), NewBufferU32(d, 1), NewBufferU32(d, uint32(wide)), NewBufferU32(d, uint32(nHhd)),
			NewBufferFloats(d, []float32{eps}), NewBufferU32(d, ao))
		got := buf.Floats()
		var maxAbs float64
		for i := range ref {
			if dd := math.Abs(float64(got[i] - ref[i])); math.IsNaN(dd) || dd > maxAbs {
				maxAbs = dd
			}
		}
		mustFinite(t, "qk_norm(whole) maxAbs", maxAbs)
		if maxAbs > 1e-4 {
			t.Fatalf("qk_norm whole-vector geometry (addOne=%v) FAIL: maxAbs=%.2e", addOne, maxAbs)
		}
		t.Logf("qk_norm whole-vector geometry (addOne=%v) vs CPU: maxAbs=%.2e — PARITY ✓", addOne, maxAbs)

		// Sanity: the whole-vector result must NOT equal the per-head result on the same input —
		// otherwise this test would pass even if encodeAttention silently kept the per-head
		// geometry (a stale bug this test exists to catch, not just a formula check in isolation).
		refPerHead := append([]float32(nil), qkv...)
		normPerHead := func(off, heads int, w []float32) {
			for h := range heads {
				b := off + h*hd
				var ss float64
				for i := range hd {
					ss += float64(refPerHead[b+i]) * float64(refPerHead[b+i])
				}
				r := 1 / math.Sqrt(ss/float64(hd)+eps)
				for i := range hd {
					wt := w[i]
					if addOne {
						wt = 1 + w[i]
					}
					refPerHead[b+i] = float32(float64(refPerHead[b+i]) * r * float64(wt))
				}
			}
		}
		normPerHead(0, nH, qn[:hd])
		normPerHead(nHhd, nH, kn[:hd])
		identical := true
		for i := range ref {
			if ref[i] != refPerHead[i] {
				identical = false
				break
			}
		}
		if identical {
			t.Fatalf("whole-vector and per-head references are identical on this random input — " +
				"the test cannot distinguish the two geometries, strengthen the input")
		}
	}
}

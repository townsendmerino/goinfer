//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestQKNorm — validates the per-head Q/K RMSNorm kernel (Qwen3/Gemma3) against a CPU reference
// that mirrors decoder/rmsnorm.go: for each head, out[i] = x[i]/sqrt(mean(x²)+eps) * (w or 1+w).
func TestQKNorm(t *testing.T) {
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
	const nH, nKV, hd = 12, 2, 128
	const eps = 1e-6
	nHhd := nH * hd
	kvDim := nKV * hd
	qkvDim := nHhd + 2*kvDim
	rng := rand.New(rand.NewSource(9))
	rvec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = rng.Float32()*2 - 1
		}
		return v
	}

	for _, addOne := range []bool{false, true} {
		qkv := rvec(qkvDim)
		qn, kn := rvec(hd), rvec(hd)
		// CPU reference (only q + k sections change; v untouched).
		ref := append([]float32(nil), qkv...)
		norm := func(off, heads int, w []float32) {
			for h := range heads {
				b := off + h*hd
				var ss float64
				for i := range hd {
					ss += float64(ref[b+i]) * float64(ref[b+i])
				}
				r := 1 / math.Sqrt(ss/float64(hd)+eps)
				for i := range hd {
					wt := w[i]
					if addOne {
						wt = 1 + w[i]
					}
					ref[b+i] = float32(float64(ref[b+i]) * r * float64(wt))
				}
			}
		}
		norm(0, nH, qn)     // q heads
		norm(nHhd, nKV, kn) // k heads (v section, off nHhd+kvDim, untouched)
		ao := uint32(0)
		if addOne {
			ao = 1
		}
		q := d.NewCommandQueue()
		buf := d.NewBufferFloats(qkv)
		q.Run1D(pipe, (nH+nKV)*128, 128, buf, d.NewBufferFloats(qn), d.NewBufferFloats(kn),
			d.NewBufferU32(nH), d.NewBufferU32(nKV), d.NewBufferU32(hd), d.NewBufferU32(uint32(nHhd)),
			d.NewBufferFloats([]float32{eps}), d.NewBufferU32(ao))
		got := buf.Floats()
		var maxAbs float64
		for i := range ref {
			if dd := math.Abs(float64(got[i] - ref[i])); math.IsNaN(dd) || dd > maxAbs {
				maxAbs = dd // propagate NaN so mustFinite can catch degenerate output
			}
		}
		mustFinite(t, "qk_norm maxAbs", maxAbs)
		if maxAbs > 1e-4 {
			t.Fatalf("qk_norm(addOne=%v) FAIL: maxAbs=%.2e", addOne, maxAbs)
		}
		t.Logf("qk_norm(addOne=%v) vs CPU: maxAbs=%.2e — PARITY ✓", addOne, maxAbs)
	}
}

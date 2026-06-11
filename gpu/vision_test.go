//go:build gpu

package gpu

import (
	"math"
	"testing"
)

// cpuLayerNorm: reference (population variance), matching HF nn.LayerNorm and
// vision.layerNorm.
func cpuLayerNorm(src, w, b []float32, rows, h int, eps float32) []float32 {
	out := make([]float32, rows*h)
	for r := 0; r < rows; r++ {
		row := src[r*h : r*h+h]
		var mean, m2 float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(h)
		for _, v := range row {
			d := float64(v) - mean
			m2 += d * d
		}
		inv := 1.0 / math.Sqrt(m2/float64(h)+float64(eps))
		for i, v := range row {
			out[r*h+i] = float32((float64(v)-mean)*inv)*w[i] + b[i]
		}
	}
	return out
}

func TestVisionLayerNorm_parity(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no gpu: %v", err)
	}
	defer c.Close()

	const rows, h = 257, 1152 // a SigLIP-ish width, an odd row count
	src := make([]float32, rows*h)
	w := make([]float32, h)
	b := make([]float32, h)
	seed := uint32(12345)
	rnd := func() float32 { seed = seed*1664525 + 1013904223; return float32(int32(seed))/float32(1<<31)*2 - 1 }
	for i := range src {
		src[i] = rnd() * 3
	}
	for i := range w {
		w[i] = rnd()
		b[i] = rnd() * 0.1
	}

	got, err := c.LayerNormRowsHost(src, w, b, rows, h, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	want := cpuLayerNorm(src, w, b, rows, h, 1e-6)
	var maxAbs float64
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("GPU LayerNorm vs CPU: max abs diff = %.3e", maxAbs)
	if maxAbs > 1e-3 {
		t.Errorf("LayerNorm GPU vs CPU max abs diff %.3e > 1e-3", maxAbs)
	}
}

func TestVisionSoftmax_parity(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no gpu: %v", err)
	}
	defer c.Close()
	const rows, n = 130, 4096
	const scale = 0.11785 // ~1/sqrt(72)
	x := make([]float32, rows*n)
	seed := uint32(99)
	rnd := func() float32 { seed = seed*1664525 + 1013904223; return float32(int32(seed))/float32(1<<31)*2 - 1 }
	for i := range x {
		x[i] = rnd() * 5
	}
	got, err := c.softmaxRowsHost(x, rows, n, scale)
	if err != nil {
		t.Fatal(err)
	}
	// CPU reference: scale, stable softmax per row.
	want := make([]float32, rows*n)
	for r := 0; r < rows; r++ {
		row := x[r*n : r*n+n]
		mx := float32(-3.4e38)
		for _, v := range row {
			if v*scale > mx {
				mx = v * scale
			}
		}
		var s float64
		for i, v := range row {
			e := math.Exp(float64(v*scale - mx))
			want[r*n+i] = float32(e)
			s += e
		}
		for i := range row {
			want[r*n+i] /= float32(s)
		}
	}
	var maxAbs, sumCheck float64
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > maxAbs {
			maxAbs = d
		}
	}
	for i := 0; i < n; i++ {
		sumCheck += float64(got[i])
	} // row 0 should sum to ~1
	t.Logf("GPU softmax vs CPU: max abs diff = %.3e, row0 sum = %.6f", maxAbs, sumCheck)
	if maxAbs > 1e-5 {
		t.Errorf("softmax max abs diff %.3e > 1e-5", maxAbs)
	}
	if math.Abs(sumCheck-1) > 1e-4 {
		t.Errorf("softmax row sum %.6f != 1", sumCheck)
	}
}

func TestVisionGelu_parity(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no gpu: %v", err)
	}
	defer c.Close()
	x := make([]float32, 5000)
	for i := range x {
		x[i] = float32(i)/500 - 5
	}
	got, err := c.geluHost(x)
	if err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for i, v := range x {
		fv := float64(v)
		want := 0.5 * fv * (1 + math.Tanh(0.7978845608028654*(fv+0.044715*fv*fv*fv)))
		if d := math.Abs(float64(got[i]) - want); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("GPU gelu-tanh vs CPU: max abs diff = %.3e", maxAbs)
	if maxAbs > 1e-4 {
		t.Errorf("gelu max abs diff %.3e > 1e-4", maxAbs)
	}
}

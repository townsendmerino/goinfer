//go:build darwin

package metal

import (
	"math"
	"testing"
)

// TestGemma4Kernels_libraryCompiles instantiates every gemma4-MoE pipeline by name — the runtime
// MSL-compile gate for gemma4MoeKernels (a syntax error fails here, not deep in a model load).
func TestGemma4Kernels_libraryCompiles(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile allKernels (incl. gemma4MoeKernels): %v", err)
	}
	for _, name := range []string{"gemv_f32_f32", "rmsnorm_nw", "scale_wgt_by_expert", "scale_vec", "zero_vec"} {
		if _, err := d.NewComputePipeline(lib, name); err != nil {
			t.Fatalf("pipeline %s: %v", name, err)
		}
	}
}

// TestGemma4Kernels_rmsnormNW isolates rmsnorm_nw (weightless OUT-OF-PLACE RMSNorm) vs a CPU oracle
// and asserts it does NOT mutate its input — the property the whole design depends on (the raw
// residual h must survive to feed the dense branch, the expert branch, and the final add). Mirrors
// cuda/gemma4_router_parity_test.go TestGemma4_rmsnormNW_scaleVec.
func TestGemma4Kernels_rmsnormNW(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "rmsnorm_nw")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	const H = 256
	const eps = float32(1e-6)
	src := make([]float32, H)
	for i := range src {
		src[i] = float32(math.Sin(float64(i)*0.17)) * 1.7
	}
	// CPU rmsNormNoWeight: x * rsqrt(mean(x^2)+eps).
	var ss float64
	for _, v := range src {
		ss += float64(v) * float64(v)
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(H)+float64(eps)))
	want := make([]float32, H)
	for i := range src {
		want[i] = src[i] * inv
	}

	srcBuf := d.NewBufferFloats(src)
	dstBuf := d.NewBufferLen(H)
	q := d.NewCommandQueue()
	q.Run1D(pipe, 256, 256, srcBuf, dstBuf, d.NewBufferU32(H), d.NewBufferFloats([]float32{eps}))
	got := dstBuf.Floats()
	gotSrc := srcBuf.Floats()

	var maxRel float64
	for i := range got {
		if r := math.Abs(float64(got[i]-want[i])) / (math.Abs(float64(want[i])) + 1e-6); r > maxRel {
			maxRel = r
		}
		if gotSrc[i] != src[i] {
			t.Fatalf("rmsnorm_nw mutated its INPUT at %d (%.6f != %.6f) — not out-of-place", i, gotSrc[i], src[i])
		}
	}
	t.Logf("rmsnorm_nw vs CPU: maxRelErr=%.2e, input preserved", maxRel)
	if maxRel > 1e-3 {
		t.Errorf("rmsnorm_nw maxRelErr %.3e > 1e-3", maxRel)
	}
}

// TestGemma4Kernels_scaleWgtByExpert isolates the on-GPU per-expert-scale fold: wgt[k] *=
// perExpertScale[idx[k]]. The generic moe_route has only a single scalar routed_scaling_factor, so
// this indexed multiply is a new op; verify it against a CPU oracle. Mirrors cuda's
// TestGemma4_perExpertScaleFold.
func TestGemma4Kernels_scaleWgtByExpert(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "scale_wgt_by_expert")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	// top-2 of 4: weights from moe_route, selected expert ids, learned per-expert scale.
	wgt := []float32{0.6, 0.4}
	idx := []uint32{2, 0}
	perExpertScale := []float32{1.5, 0.5, 2.0, 1.0}
	K := len(wgt)
	want := make([]float32, K)
	for k := 0; k < K; k++ {
		want[k] = wgt[k] * perExpertScale[idx[k]] // 0.6*2.0=1.2 ; 0.4*1.5=0.6
	}
	wgtBuf := d.NewBufferFloats(wgt)
	q := d.NewCommandQueue()
	q.Run1D(pipe, K, K, wgtBuf, d.NewBufferUint32s(idx), d.NewBufferFloats(perExpertScale), d.NewBufferU32(uint32(K)))
	got := wgtBuf.Floats()
	for k := 0; k < K; k++ {
		if dd := math.Abs(float64(got[k] - want[k])); dd > 1e-6 {
			t.Errorf("wgt[%d]: got %.6f want %.6f (idx=%d scale=%.3f)", k, got[k], want[k], idx[k], perExpertScale[idx[k]])
		}
	}
	t.Logf("per-expert-scale fold: %v * scale[%v] = %v (matches CPU)", wgt, idx, got)
}

// TestGemma4Kernels_scaleVec_zeroVec isolates the two remaining elementwise kernels: scale_vec
// (x *= s, the per-layer output scalar) and zero_vec (x = 0, the expert-accumulator clear that must
// survive a stale NaN — a multiply-by-zero would not).
func TestGemma4Kernels_scaleVec_zeroVec(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pScale, err := d.NewComputePipeline(lib, "scale_vec")
	if err != nil {
		t.Fatalf("pipeline scale_vec: %v", err)
	}
	pZero, err := d.NewComputePipeline(lib, "zero_vec")
	if err != nil {
		t.Fatalf("pipeline zero_vec: %v", err)
	}
	const H = 64
	x := make([]float32, H)
	for i := range x {
		x[i] = float32(math.Cos(float64(i)*0.3)) * 2.1
	}
	const s = 0.375
	q := d.NewCommandQueue()

	xBuf := d.NewBufferFloats(x)
	q.Run1D(pScale, H, 256, xBuf, d.NewBufferFloats([]float32{s}))
	gotScale := xBuf.Floats()
	for i := range x {
		if dd := math.Abs(float64(gotScale[i] - x[i]*s)); dd > 1e-6 {
			t.Fatalf("scale_vec[%d]=%.6f want %.6f", i, gotScale[i], x[i]*s)
		}
	}

	// zero_vec must zero even a NaN-poisoned buffer (the stale-scratch case).
	poison := make([]float32, H)
	nan := float32(math.NaN())
	for i := range poison {
		poison[i] = nan
	}
	zBuf := d.NewBufferFloats(poison)
	q.Run1D(pZero, H, 256, zBuf)
	gotZero := zBuf.Floats()
	for i := range gotZero {
		if gotZero[i] != 0 {
			t.Fatalf("zero_vec[%d]=%v (not zero — a NaN survived)", i, gotZero[i])
		}
	}
	t.Logf("scale_vec (x*%.3f) and zero_vec (NaN→0) match CPU", s)
}

//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestSAGemvBiasResid gates gemv_w4a8_sa_bias_resid (kernels.go) — FeatOutBias's kernel: the
// attention output projection for a family with an additive o_proj bias (GPT-2, gpt-oss) needs
// BOTH the residual accumulation gemv_w4a8_sa_resid already does AND the per-row bias
// gemv_w4a8_sa_bias already does, and no existing kernel does both. Reference: CPU dequant of
// the same packed int4 weights/int8 activation (packW4A8Row, the validated packer other SA
// tests already use), plus bias, accumulated onto a nonzero pre-existing residual — so the test
// would fail if either the bias OR the accumulation were silently dropped, not just if both
// were.
func TestSAGemvBiasResid(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w4a8_sa_bias_resid")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const N, K = 128, 512
	rng := rand.New(rand.NewSource(31))

	a := make([]float32, K)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}
	amx := float32(0)
	for _, v := range a {
		if x := float32(math.Abs(float64(v))); x > amx {
			amx = x
		}
	}
	aSc := amx / 127
	aq := make([]int8, K)
	for i, v := range a {
		aq[i] = int8(math.Max(-127, math.Min(127, math.Round(float64(v/aSc)))))
	}

	bias := make([]float32, N)
	resid0 := make([]float32, N)
	for n := range N {
		bias[n] = rng.Float32()*2 - 1
		resid0[n] = rng.Float32()*4 - 2 // a nonzero pre-existing residual to accumulate onto
	}

	words := make([]uint32, N*(K/8))
	scalesH := make([]uint16, N*(K/32))
	ref := make([]float32, N)
	for n := range N {
		row := make([]float32, K)
		for k := range row {
			row[k] = rng.Float32()*2 - 1
		}
		w, s := packW4A8Row(row)
		copy(words[n*(K/8):(n+1)*(K/8)], w)
		var acc float64
		for g := range K / 32 {
			scalesH[n*(K/32)+g] = f32ToF16(s[g])
			sc := float64(f16ToF32(scalesH[n*(K/32)+g]))
			for e := range 32 {
				k := g*32 + e
				nib := int((w[k/8]>>(4*uint(k%8)))&0xF) - 8
				acc += float64(nib) * float64(aq[k]) * sc
			}
		}
		ref[n] = resid0[n] + float32(acc)*aSc + bias[n]
	}

	q := d.NewCommandQueue()
	out := d.NewBufferFloats(resid0)
	q.Run1DTG(pipe, N*32, 256, K*2,
		d.NewBufferUint32s(words), d.NewBufferU16s(scalesH), d.NewBufferInt8(aq),
		d.NewBufferFloats([]float32{aSc}), out, d.NewBufferFloats(bias), d.NewBufferU32(K))
	got := out.Floats()

	var dot, na, nb, maxRel float64
	for n := range N {
		dot += float64(got[n]) * float64(ref[n])
		na += float64(got[n]) * float64(got[n])
		nb += float64(ref[n]) * float64(ref[n])
		if dd := math.Abs(float64(got[n] - ref[n])); dd > 1e-3 {
			if rel := dd / (math.Abs(float64(ref[n])) + 1e-3); rel > maxRel {
				maxRel = rel
			}
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	mustFinite(t, "SA bias+resid GEMV cosine", cos)
	if cos < 0.9999 || maxRel > 5e-3 {
		t.Fatalf("gemv_w4a8_sa_bias_resid parity FAIL: cos=%.7f maxRel=%.4f", cos, maxRel)
	}
	t.Logf("gemv_w4a8_sa_bias_resid N=%d K=%d vs CPU: cos=%.7f maxRel=%.4f — PARITY ✓", N, K, cos, maxRel)

	// Isolate a bias-dropped or residual-dropped regression from a coincidental pass: check
	// against the two WRONG references directly, so future readers see exactly what broke.
	biasOnly := make([]float32, N)  // no bias, has residual
	residOnly := make([]float32, N) // has bias, no residual
	for n := range N {
		biasOnly[n] = ref[n] - bias[n]
		residOnly[n] = ref[n] - resid0[n]
	}
	if closeEnough(got, biasOnly, 1e-2) {
		t.Error("output matches a bias-DROPPED reference — the bias epilogue is not being applied")
	}
	if closeEnough(got, residOnly, 1e-2) {
		t.Error("output matches a residual-DROPPED reference — the accumulation is not happening")
	}
}

// TestCoalGemvBiasResid gates gemv_w4a8_resid_bias — the coal-family counterpart to
// gemv_w4a8_sa_bias_resid, needed for GPT-2's FFN down-projection (K=intermediate=3072 exceeds
// the SA family's 1536 cap, so down-proj uses the coal family same as every other wide
// projection). Same reference/negative-check structure as TestSAGemvBiasResid; only the launch
// geometry differs (N*32 threads / tg=32, no dynamic threadgroup memory — W4A8_BODY does its
// unpack per-lane, not staged like SA_BODY).
func TestCoalGemvBiasResid(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "gemv_w4a8_resid_bias")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const N, K = 64, 3072 // GPT-2 small's down-proj: out=hidden(768)... use N=64 rows, K=intermediate
	rng := rand.New(rand.NewSource(37))

	a := make([]float32, K)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}
	amx := float32(0)
	for _, v := range a {
		if x := float32(math.Abs(float64(v))); x > amx {
			amx = x
		}
	}
	aSc := amx / 127
	aq := make([]int8, K)
	for i, v := range a {
		aq[i] = int8(math.Max(-127, math.Min(127, math.Round(float64(v/aSc)))))
	}

	bias := make([]float32, N)
	resid0 := make([]float32, N)
	for n := range N {
		bias[n] = rng.Float32()*2 - 1
		resid0[n] = rng.Float32()*4 - 2
	}

	words := make([]uint32, N*(K/8))
	scalesH := make([]uint16, N*(K/32))
	ref := make([]float32, N)
	for n := range N {
		row := make([]float32, K)
		for k := range row {
			row[k] = rng.Float32()*2 - 1
		}
		w, s := packW4A8Row(row)
		copy(words[n*(K/8):(n+1)*(K/8)], w)
		var acc float64
		for g := range K / 32 {
			scalesH[n*(K/32)+g] = f32ToF16(s[g])
			sc := float64(f16ToF32(scalesH[n*(K/32)+g]))
			for e := range 32 {
				k := g*32 + e
				nib := int((w[k/8]>>(4*uint(k%8)))&0xF) - 8
				acc += float64(nib) * float64(aq[k]) * sc
			}
		}
		ref[n] = resid0[n] + float32(acc)*aSc + bias[n]
	}

	q := d.NewCommandQueue()
	out := d.NewBufferFloats(resid0)
	q.Run1D(pipe, N*32, 32,
		d.NewBufferUint32s(words), d.NewBufferU16s(scalesH), d.NewBufferInt8(aq),
		d.NewBufferFloats([]float32{aSc}), out, d.NewBufferFloats(bias), d.NewBufferU32(K))
	got := out.Floats()

	var dot, na, nb, maxRel float64
	for n := range N {
		dot += float64(got[n]) * float64(ref[n])
		na += float64(got[n]) * float64(got[n])
		nb += float64(ref[n]) * float64(ref[n])
		if dd := math.Abs(float64(got[n] - ref[n])); dd > 1e-3 {
			if rel := dd / (math.Abs(float64(ref[n])) + 1e-3); rel > maxRel {
				maxRel = rel
			}
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	mustFinite(t, "coal bias+resid GEMV cosine", cos)
	if cos < 0.9999 || maxRel > 5e-3 {
		t.Fatalf("gemv_w4a8_resid_bias parity FAIL: cos=%.7f maxRel=%.4f", cos, maxRel)
	}
	t.Logf("gemv_w4a8_resid_bias N=%d K=%d vs CPU: cos=%.7f maxRel=%.4f — PARITY ✓", N, K, cos, maxRel)

	biasOnly := make([]float32, N)
	residOnly := make([]float32, N)
	for n := range N {
		biasOnly[n] = ref[n] - bias[n]
		residOnly[n] = ref[n] - resid0[n]
	}
	if closeEnough(got, biasOnly, 1e-2) {
		t.Error("output matches a bias-DROPPED reference — the bias epilogue is not being applied")
	}
	if closeEnough(got, residOnly, 1e-2) {
		t.Error("output matches a residual-DROPPED reference — the accumulation is not happening")
	}
}

func closeEnough(a, b []float32, tol float64) bool {
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > tol {
			return false
		}
	}
	return true
}

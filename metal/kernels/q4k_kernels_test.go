//go:build darwin

package kernels

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestQ4KLaneKernels is G-Q1 of docs/tasks/task-metal-q4k-2026-10.md (registered before this code), and since KS-C1 its C1-G1
// (docs/tasks/task-kernel-survey-2026-10.md): the q4k lane's two GEMVs (the rows kernels at every built R; plain, residual and
// bias variants) against a float64 reference on the host, which dequantizes the same bytes through aikit (linalg.WrapQ4K's Row for
// Q4_K; the per-row scale for int8) and dots them with the same f32 activations. Bar: per row, |got - ref| at most 1e-5
// of Σ|w·a| (f32 accumulation order only). Shapes: every Phi-3 projection's (o, gate|up, down, qkv, the LM head) and
// edge cases (one super-block, a partial threadgroup). Each registered planted defect, compiled into its own copy of the
// kernels, must miss the bar.
func TestQ4KLaneKernels(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	rng := rand.New(rand.NewSource(11))
	type shape struct {
		name string
		N, K int
	}
	q4kShapes := []shape{{"one super-block", 1, 256}, {"partial threadgroup", 7, 512}, {"o", 3072, 3072},
		{"gate|up", 16384, 3072}, {"down", 3072, 8192}}
	w8Shapes := []shape{{"partial threadgroup", 5, 64}, {"qkv", 9216, 3072}, {"down (Q6_K layers)", 3072, 8192},
		{"lm head", 32064, 3072}}

	acts := func(K int) []float32 {
		a := make([]float32, K)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		for i := 0; i < K; i += 997 { // Phi-3-like outliers: what per-row int8 activations round the rest to zero over
			a[i] = float32(300 * rng.NormFloat64())
		}
		return a
	}
	q4kRaw := func(N, K int) []byte {
		raw := make([]byte, N*linalg.Q4KRowBytes(K))
		for b := 0; b < len(raw); b += 144 {
			putF16(raw[b:], float32(0.002+0.01*rng.Float64()))
			putF16(raw[b+2:], float32(0.001+0.005*rng.Float64()))
			for i := 4; i < 144; i++ {
				raw[b+i] = byte(rng.Intn(256))
			}
			if b%(144*3) == 0 { // the 6-bit range ends: every scale and min at 63, then at 0
				for i := 4; i < 16; i++ {
					raw[b+i] = 0xFF
				}
			}
		}
		return raw
	}
	// check runs src's kernels on every shape and returns the worst error as a fraction of Σ|w·a| over all rows.
	check := func(src string) (worstQ4K, worstW8 float64) {
		lib, err := d.CompileLibrary(src, MSL3_1)
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		pipe := func(name string) Pipeline {
			p, err := d.NewComputePipeline(lib, name)
			if err != nil {
				t.Fatal(err)
			}
			return p
		}
		biases := func(N int) []float32 {
			b := make([]float32, N)
			for i := range b {
				b[i] = float32(rng.NormFloat64())
			}
			return b
		}
		q := d.NewCommandQueue()
		rowsRef := func(wm *linalg.WeightMat, a []float32, N, K int) ([]float64, []float64) {
			ref, mag := make([]float64, N), make([]float64, N)
			row := make([]float32, K)
			for n := range N {
				wm.Row(n, row)
				for k := range K {
					ref[n] += float64(row[k]) * float64(a[k])
					mag[n] += math.Abs(float64(row[k]) * float64(a[k]))
				}
			}
			return ref, mag
		}
		for _, sh := range q4kShapes {
			raw := q4kRaw(sh.N, sh.K)
			wm, err := linalg.WrapQ4K(raw, sh.N, sh.K)
			if err != nil {
				t.Fatal(err)
			}
			a := acts(sh.K)
			ref, mag := rowsRef(&wm, a, sh.N, sh.K)
			bW, bA, bOut := NewBufferInt8(d, asInt8(raw)), NewBufferFloats(d, a), d.NewBufferLen(sh.N)
			init := make([]float32, sh.N)
			for i := range init {
				init[i] = float32(i % 7)
			}
			bRes := NewBufferFloats(d, init)
			bias := biases(sh.N)
			bBias, bOutB := NewBufferFloats(d, bias), d.NewBufferLen(sh.N)
			uK, uN := NewBufferU32(d, uint32(sh.K)), NewBufferU32(d, uint32(sh.N))
			for _, R := range []int{1, 2, 4, 8} {
				n := (sh.N + R - 1) / R * 32
				copy(bRes.Floats()[:sh.N], init)
				e := q.Begin()
				e.Dispatch(pipe(fmt.Sprintf("gemv_q4k_f32_rows%d", R)), n, 256, bW, bA, bOut, uK, uN)
				e.Dispatch(pipe(fmt.Sprintf("gemv_q4k_f32_resid_rows%d", R)), n, 256, bW, bA, bRes, uK, uN)
				e.Dispatch(pipe(fmt.Sprintf("gemv_q4k_f32_bias_rows%d", R)), n, 256, bW, bA, bOutB, uK, uN, bBias)
				e.End()
				got, gotRes, gotB := bOut.Floats(), bRes.Floats(), bOutB.Floats()
				for n := range sh.N {
					worstQ4K = math.Max(worstQ4K, math.Abs(float64(got[n])-ref[n])/math.Max(mag[n], 1e-30))
					worstQ4K = math.Max(worstQ4K, math.Abs(float64(gotRes[n])-(ref[n]+float64(init[n])))/math.Max(mag[n], 1e-30))
					worstQ4K = math.Max(worstQ4K, math.Abs(float64(gotB[n])-(ref[n]+float64(bias[n])))/math.Max(mag[n], 1e-30))
				}
			}
		}
		for _, sh := range w8Shapes {
			w8 := make([]int8, sh.N*sh.K)
			for i := range w8 {
				w8[i] = int8(rng.Intn(255) - 127)
			}
			sc := make([]float32, sh.N)
			for i := range sc {
				sc[i] = float32(0.001 + 0.01*rng.Float64())
			}
			a := acts(sh.K)
			ref, mag := make([]float64, sh.N), make([]float64, sh.N)
			for n := range sh.N {
				for k := range sh.K {
					v := float64(w8[n*sh.K+k]) * float64(sc[n]) * float64(a[k])
					ref[n] += v
					mag[n] += math.Abs(v)
				}
			}
			bW, bS, bA, bOut := NewBufferInt8(d, w8), NewBufferFloats(d, sc), NewBufferFloats(d, a), d.NewBufferLen(sh.N)
			bias := biases(sh.N)
			bBias, bOutB := NewBufferFloats(d, bias), d.NewBufferLen(sh.N)
			uK, uN := NewBufferU32(d, uint32(sh.K)), NewBufferU32(d, uint32(sh.N))
			for _, R := range []int{1, 2, 4} {
				n := (sh.N + R - 1) / R * 32
				e := q.Begin()
				e.Dispatch(pipe(fmt.Sprintf("gemv_w8_f32_rows%d", R)), n, 256, bW, bS, bA, bOut, uK, uN)
				e.Dispatch(pipe(fmt.Sprintf("gemv_w8_f32_bias_rows%d", R)), n, 256, bW, bS, bA, bOutB, uK, uN, bBias)
				e.End()
				got, gotB := bOut.Floats(), bOutB.Floats()
				for n := range sh.N {
					worstW8 = math.Max(worstW8, math.Abs(float64(got[n])-ref[n])/math.Max(mag[n], 1e-30))
					worstW8 = math.Max(worstW8, math.Abs(float64(gotB[n])-(ref[n]+float64(bias[n])))/math.Max(mag[n], 1e-30))
				}
			}
		}
		return worstQ4K, worstW8
	}

	const bar = 1e-5
	wq, ww := check(AllKernels)
	fmt.Printf("[G-Q1] shipped kernels: worst |err|/Σ|w·a| %.2e (q4k), %.2e (int8)\n", wq, ww)
	if wq > bar || ww > bar {
		t.Errorf("G-Q1 FAIL: worst error %.2e (q4k) / %.2e (int8), bar %.0e", wq, ww, bar)
	}
	defects := []struct{ name, from, to string }{
		{"(1) the min term dropped", "acc[r] += d*sc0*dlo + d*sc1*dhi - dm*(m0*slo + m1*shi);", "acc[r] += d*sc0*dlo + d*sc1*dhi;"},
		{"(2) the 6-bit scale and min unpack swapped", "if (j < 4u) { sc = float(q[j] & 63u); m = float(q[j+4u] & 63u); }",
			"if (j < 4u) { m = float(q[j] & 63u); sc = float(q[j+4u] & 63u); }"},
		{"(3) a super-block offset of one", "device const uchar* blk = wq + (ulong)(row0 + r)*(ulong)NB*144ul + b*144u;",
			"device const uchar* blk = wq + (ulong)(row0 + r)*(ulong)NB*144ul + ((b+1u)%NB)*144u;"},
		{"(4) the int8 row scale dropped", "acc[r] = simd_sum(acc[r]) * sc[min(row0 + r, N - 1u)];", "acc[r] = simd_sum(acc[r]);"},
		{"(5) KS-C1: the high half's min term dropped", "- dm*(m0*slo + m1*shi);", "- dm*(m0*slo);"},
	}
	for _, df := range defects {
		if !strings.Contains(AllKernels, df.from) {
			t.Fatalf("planted defect %s: its source text is gone from the kernels", df.name)
		}
		wq, ww := check(strings.Replace(AllKernels, df.from, df.to, 1))
		fmt.Printf("[G-Q1] planted %s: worst %.2e (q4k), %.2e (int8)\n", df.name, wq, ww)
		if wq <= bar && ww <= bar {
			t.Errorf("planted defect %s left the bar green", df.name)
		}
	}
}

// asInt8 is raw's bytes as int8, for NewBufferInt8 (the same bits on the device).
func asInt8(raw []byte) []int8 {
	out := make([]int8, len(raw))
	for i, b := range raw {
		out[i] = int8(b)
	}
	return out
}

// putF16 writes v as a little-endian IEEE half.
func putF16(b []byte, v float32) {
	h := F32ToF16(v)
	b[0], b[1] = byte(h), byte(h>>8)
}

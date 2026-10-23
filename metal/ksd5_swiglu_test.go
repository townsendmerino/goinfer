//go:build darwin

package metal

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
)

// TestKSD5_fusedSwigluMatchesTwoStep is D5-G1's kernel half (docs/tasks/task-kernel-survey-2026-10.md, "KS-D5 gates"):
// gemm_w4f16_swiglu against gemm_w4f16_store over the gate|up rows followed by swiglu_f16, at random int4 weights, f16 scales and
// activations; M = 64, 100, 200; I = 96 and 8,960 (K = 1,536, the 1.5B's hidden); SiLU and GELU-tanh. Every output byte-equal.
func TestKSD5_fusedSwigluMatchesTwoStep(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(n string) Pipeline {
		p, e := d.NewComputePipeline(lib, n)
		if e != nil {
			t.Fatalf("%s: %v", n, e)
		}
		return p
	}
	pGemm, pSw, pFused := pipe("gemm_w4f16_store"), pipe("swiglu_f16"), pipe("gemm_w4f16_swiglu")
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(13))
	const K = 1536
	for _, I := range []int{96, 8960} {
		N := 2 * I
		words := make([]uint32, N*K/8)
		for i := range words {
			words[i] = rng.Uint32()
		}
		scales := make([]uint16, N*K/32)
		for i := range scales {
			scales[i] = f32ToF16(float32(rng.NormFloat64() * 0.02))
		}
		W, WS := NewBufferUint32s(d, words), NewBufferU16s(d, scales)
		for _, M := range []int{64, 100, 200} {
			a := make([]uint16, M*K)
			for i := range a {
				a[i] = f32ToF16(float32(rng.NormFloat64()))
			}
			A := NewBufferU16s(d, a)
			for _, act := range []uint32{1, 0} {
				gu, out, fused := NewBufferU16s(d, make([]uint16, M*N)), NewBufferU16s(d, make([]uint16, M*I)), NewBufferU16s(d, make([]uint16, M*I))
				uM, uN, uI, uK, uAct, uMode := NewBufferU32(d, uint32(M)), NewBufferU32(d, uint32(N)), NewBufferU32(d, uint32(I)), NewBufferU32(d, K), NewBufferU32(d, act), NewBufferU32(d, 0)
				bias := NewBufferFloats(d, make([]float32, N))
				e := q.Begin()
				e.Dispatch2D(pGemm, (N+63)/64, (M+63)/64, 128, 1, A, W, WS, gu, uM, uN, uK, bias, uMode)
				e.Dispatch(pSw, M*I, 256, gu, out, uI, uAct)
				e.Dispatch2D(pFused, (I+31)/32, (M+63)/64, 128, 1, A, W, WS, fused, uM, uI, uK, uAct)
				e.End()
				want, got := out.U16s()[:M*I], fused.U16s()[:M*I]
				diff := 0
				for i := range want {
					if want[i] != got[i] {
						diff++
					}
				}
				fmt.Fprintf(os.Stderr, "[D5-G1] I=%d M=%d act=%d: %d of %d outputs differ\n", I, M, act, diff, M*I)
				if diff != 0 {
					t.Errorf("I=%d M=%d act=%d: %d outputs differ from GEMM + swiglu_f16", I, M, act, diff)
				}
			}
		}
	}
}

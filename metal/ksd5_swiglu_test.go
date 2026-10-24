//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
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

// TestKSD5c_w8FusedSwigluMatchesTwoStep is D5c-G1's kernel half (docs/tasks/task-kernel-survey-2026-10.md, "KS-D5c gates"):
// gemm_w8f16_swiglu against gemm_w8f16_m64n64 over the gate|up rows followed by swiglu_f16, at random int8 codes, f32 row scales
// and f16 activations, D5-G1's 12 cells, in the fast-math library and in the precise one a native int8 resident compiles.
// Every output byte-equal.
func TestKSD5c_w8FusedSwigluMatchesTwoStep(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	for _, precise := range []bool{false, true} {
		compile := d.CompileLibrary
		if precise {
			compile = d.CompileLibraryPrecise
		}
		lib, err := compile(prefillKernels, MSL3_1)
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
		pGemm, pSw, pFused := pipe("gemm_w8f16_m64n64"), pipe("swiglu_f16"), pipe("gemm_w8f16_swiglu")
		q := d.NewCommandQueue()
		rng := rand.New(rand.NewSource(17))
		const K = 1536
		for _, I := range []int{96, 8960} {
			N := 2 * I
			words := make([]uint32, N*K/4) // N rows of K int8 codes
			for i := range words {
				words[i] = rng.Uint32()
			}
			scales := make([]float32, N)
			for i := range scales {
				scales[i] = float32(rng.NormFloat64()*0.002 + 0.004)
			}
			W, WS := NewBufferUint32s(d, words), NewBufferFloats(d, scales)
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
					fmt.Fprintf(os.Stderr, "[D5c-G1] precise=%v I=%d M=%d act=%d: %d of %d outputs differ\n", precise, I, M, act, diff, M*I)
					if diff != 0 {
						t.Errorf("precise=%v I=%d M=%d act=%d: %d outputs differ from the W8 GEMM + swiglu_f16", precise, I, M, act, diff)
					}
				}
			}
		}
	}
}

// ksd5Checkpoint is the real model the KS-D5 model tests load: the 1.5B's int4 Metal .giw, or with GOINFER_KSD5_QUANT=int8int8
// the 0.5B's .gguf at int8int8 (the native int8 lane, KS-D5c; the 1.5B at int8 does not fit beside the owner's apps), loaded
// directly so nothing is written beside it.
func ksd5Checkpoint() (string, decoder.Options) {
	if os.Getenv("GOINFER_KSD5_QUANT") == "int8int8" {
		return os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
			decoder.Options{Backend: "metal", Quant: "int8int8", DirectLoad: true, ResidentContext: 4096}
	}
	return os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v17.giw"), decoder.Options{Backend: "metal", ResidentContext: 4096}
}

// TestKSD5_prefillFusedIdentity is D5-G1's and D5c-G1's model half: llama-tiny at int4 and at int8int8 (the native int8 lane)
// and, with GOINFER_HEAVY_TESTS=1, a real model at 3,900 rows (ksd5Checkpoint), a batched prefill's last logits with the
// gate|up GEMM and SwiGLU fused against the two-step form, byte-equal, on one resident; the pass must have taken the fused form
// (no guF counted), and the int8int8 loads the native int8 lane.
func TestKSD5_prefillFusedIdentity(t *testing.T) {
	cases := []struct {
		path string
		opts decoder.Options
		n    int
	}{
		{"../testdata/llama-tiny", decoder.Options{Backend: "metal", Quant: "int4"}, 100},
		{"../testdata/llama-tiny", decoder.Options{Backend: "metal", Quant: "int8int8"}, 100},
	}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		path, opts := ksd5Checkpoint()
		cases = append(cases, struct {
			path string
			opts decoder.Options
			n    int
		}{path, opts, 3900})
	}
	for _, c := range cases {
		m, err := decoder.Load(c.path, c.opts)
		if err != nil {
			t.Fatalf("load %s: %v", c.path, err)
		}
		decoder.SetKnobEnvForTest(t, m, "GOINFER_METAL_FAST_PREFILL_FLOOR", "0")
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("%s: not Metal-resident: %s", c.path, m.ResidentDecline())
		}
		if want := c.opts.Quant == "int8int8"; a.r.w8 != want {
			t.Fatalf("%s at %q: native int8 lane %v, want %v", c.path, c.opts.Quant, a.r.w8, want)
		}
		_, _, _, _, _, _, vocab := m.Dims()
		embs := make([][]float32, c.n)
		for i := range embs {
			embs[i] = m.EmbedResidentForTest((i*7919 + 13) % vocab)
		}
		run := func(off bool) ([]float32, int64) {
			prev := swigluFusedOffForTest
			swigluFusedOffForTest = off
			defer func() { swigluFusedOffForTest = prev }()
			a.Reset()
			before := prefillScratchSeen.Load()
			lg, err := a.PrefillLast(context.Background(), embs, 0)
			if err != nil {
				t.Fatal(err)
			}
			return append([]float32(nil), lg...), prefillScratchSeen.Load() - before
		}
		want, sTwo := run(true)
		got, sFused := run(false)
		d := 0
		for i := range want {
			if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
				d++
			}
		}
		fmt.Fprintf(os.Stderr, "[D5-G1] %s (%s, w8 %v), %d rows: %d of %d logits differ; scratch counted two-step %d B, fused %d B (fused here: %v)\n",
			c.path, c.opts.Quant, a.r.w8, c.n, d, len(want), sTwo, sFused, a.r.swigluFused((c.n+7)/8*8))
		if d != 0 {
			t.Errorf("%s: %d logits differ between the fused and two-step forms", c.path, d)
		}
		if !a.r.swigluFused((c.n+7)/8*8) || sFused >= sTwo {
			t.Errorf("%s: the pass did not take the fused form (scratch %d vs %d B)", c.path, sFused, sTwo)
		}
		m.Close()
	}
}

// TestKSD5_prefillSpeed is D5-G3 (and D5c-G3 with GOINFER_KSD5_QUANT=int8int8): a real model (ksd5Checkpoint), a 3,900-token PrefillLast, the fused gate|up GEMM against
// the two-step form on one resident (the choice is read per pass), A B B A over 10 pairs after 2 warm runs each. A timed run:
// hold the timing lock (GOINFER_KSD5_TIMING=1).
func TestKSD5_prefillSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSD5_TIMING") == "" {
		t.Skip("set GOINFER_KSD5_TIMING=1 (a timed run under the timing lock)")
	}
	path, opts := ksd5Checkpoint()
	m, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	fmt.Fprintf(os.Stderr, "[D5-G3] %s (%q, native int8 lane %v)\n", path, opts.Quant, a.r.w8)
	_, _, _, _, _, _, vocab := m.Dims()
	embs := make([][]float32, 3900)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest((i*7919 + 13) % vocab)
	}
	run := func(off bool) float64 {
		prev := swigluFusedOffForTest
		swigluFusedOffForTest = off
		defer func() { swigluFusedOffForTest = prev }()
		a.Reset()
		t0 := time.Now()
		if _, err := a.PrefillLast(context.Background(), embs, 0); err != nil {
			t.Fatal(err)
		}
		return time.Since(t0).Seconds()
	}
	for range 2 {
		run(true)
		run(false)
	}
	var ratios []float64
	below := 0
	for i := range 10 {
		var two, fused float64
		if i%2 == 0 {
			two, fused = run(true), run(false)
		} else {
			fused, two = run(false), run(true)
		}
		ratios = append(ratios, fused/two)
		if fused < two {
			below++
		}
		fmt.Fprintf(os.Stderr, "[D5-G3] pair %d: two-step %.1f ms, fused %.1f ms, fused/two-step %.4f\n", i, two*1e3, fused*1e3, fused/two)
	}
	slices.Sort(ratios)
	fmt.Fprintf(os.Stderr, "[D5-G3] median fused/two-step %.4f, %d of 10 pairs below 1\n", (ratios[4]+ratios[5])/2, below)
}

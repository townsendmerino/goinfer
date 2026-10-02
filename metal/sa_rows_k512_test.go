//go:build darwin

package metal

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// TestSARowsK512_bitIdentical is B-P04's kernel gate (docs/audit-metal-2026-09-30.md): each lane-balanced SA rows
// kernel (sa_rows_acc_k512) against the rows kernel it replaces, at every K % 512 == 0 shape from one trip to eight,
// R = 2 and 4, all three epilogues (plain, +bias, +=residual), on random nibbles, signed f16 scales (subnormals and
// large values included) and full-range int8 activations. Every output must equal the replaced kernel's bit for bit:
// the balanced form only moves where each group's exact integer sums are formed, never the float chain.
func TestSARowsK512_bitIdentical(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(name string) Pipeline {
		p, err := d.NewComputePipeline(lib, name)
		if err != nil {
			t.Fatalf("pipeline %s: %v", name, err)
		}
		return p
	}
	cq := d.NewCommandQueue()
	seed := uint32(51215)
	rnd := func() uint32 { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed }
	const N = 128 // rows: whole threadgroups of 8 simdgroups x R rows at R = 2 and 4
	checked := 0
	for _, K := range []int{512, 1024, 1536, 2048, 3584, 4096} {
		G := K / 32
		w := d.NewBufferLen(N * K / 8)
		wv := w.U32s()[:N*K/8]
		for i := range wv {
			wv[i] = rnd()
		}
		sc := make([]uint16, N*G)
		for i := range sc {
			switch rnd() % 8 {
			case 0:
				sc[i] = uint16(rnd()%1023 + 1) // a subnormal
			case 1:
				sc[i] = f32ToF16(-float32(rnd()%60000) * 1e-3)
			default:
				sc[i] = f32ToF16(float32(int(rnd()%2001)-1000) * 1e-5)
			}
		}
		s := NewBufferU16s(d, sc)
		aq := d.NewBufferBytes(K)
		av := aq.Int8s()[:K]
		for i := range av {
			av[i] = int8(rnd())
		}
		bias := make([]float32, N)
		init := make([]float32, N)
		for i := range bias {
			bias[i] = float32(int(rnd()%2001)-1000) * 1e-3
			init[i] = float32(int(rnd()%2001)-1000) * 1e-2
		}
		asc, bB, uK := NewBufferFloats(d, []float32{0.0173}), NewBufferFloats(d, bias), NewBufferU32(d, uint32(K))
		for _, R := range []int{2, 4} {
			for _, kind := range []string{"gemv_w4a8_sa_rows", "gemv_w4a8_sa_bias_rows", "gemv_w4a8_sa_resid_rows"} {
				run := func(name string) []float32 {
					out := NewBufferFloats(d, init) // the residual epilogue adds into it
					args := []Buffer{w, s, aq, asc, out, uK}
					if kind == "gemv_w4a8_sa_bias_rows" {
						args = []Buffer{w, s, aq, asc, out, bB, uK}
					}
					e := cq.Begin()
					e.DispatchTG(pipe(name), N*32/R, 256, K*2, args...)
					e.End()
					if err := e.Err(); err != nil {
						t.Fatalf("%s K=%d: %v", name, K, err)
					}
					return append([]float32(nil), out.Floats()[:N]...)
				}
				ref := run(fmt.Sprintf("%s%d", kind, R))
				got := run(fmt.Sprintf("%s_k512_%d", kind, R))
				diff := 0
				for i := range ref {
					if math.Float32bits(ref[i]) != math.Float32bits(got[i]) {
						diff++
					}
				}
				if diff != 0 {
					t.Errorf("%s_k512_%d at K=%d: %d of %d outputs differ from %s%d", kind, R, K, diff, N, kind, R)
				}
				checked += N
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[sa-k512] %d outputs across K = 512..4096, R = 2/4, three epilogues: all bit-identical to the replaced kernels\n", checked)
}

// TestSARowsK512_decodeBitIdentical is B-P04 through its caller: a real checkpoint (GOINFER_METAL_MC3=1, the 1.5B by
// default; GOINFER_METAL_MC3_MODEL for the 7B) decodes the same teacher-forced 40 tokens through the production executor
// twice, built once with the lane-balanced SA rows kernels (the default where K % 512 == 0) and once with saK512Off.
// Every logit of every position must match bit for bit, and the default build must have taken the balanced kernels,
// or the comparison is of a kernel against itself.
func TestSARowsK512_decodeBitIdentical(t *testing.T) {
	if os.Getenv("GOINFER_METAL_MC3") != "1" {
		t.Skip("set GOINFER_METAL_MC3=1 (loads a real checkpoint)")
	}
	const n = 40
	decode := func(t *testing.T, off bool) [][]float32 {
		saK512Off = off
		defer func() { saK512Off = false }()
		_, r := mc3LoadCheckpoint(t, 1, 1024)
		if r.saK512 == off {
			t.Fatalf("saK512Off=%v built saK512=%v (H %d): the arm is not the kernel it claims", off, r.saK512, r.H)
		}
		var out [][]float32
		for i := range n {
			lg := r.ForwardEmbPipe(mc3Emb(r, (i*977+31)%r.V), i)
			if err := r.takeExecErr(); err != nil {
				t.Fatalf("decode at %d: %v", i, err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	var bal, prev [][]float32
	t.Run("balanced", func(t *testing.T) { bal = decode(t, false) })
	t.Run("previous", func(t *testing.T) { prev = decode(t, true) })
	if len(bal) != n || len(prev) != n {
		t.Fatal("an arm did not run")
	}
	pos, worst := 0, 0
	for i := range bal {
		diff := 0
		for j := range bal[i] {
			if math.Float32bits(bal[i][j]) != math.Float32bits(prev[i][j]) {
				diff++
			}
		}
		if diff > 0 {
			pos++
		}
		worst = max(worst, diff)
	}
	fmt.Fprintf(os.Stderr, "[sa-k512] decode, %d teacher-forced positions through the executor: %d positions differ, worst %d of %d logits\n", n, pos, worst, len(bal[0]))
	if pos != 0 {
		t.Errorf("%d of %d positions differ between the balanced and previous SA rows kernels", pos, n)
	}
}

//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSD8_seqKernelsMatchDecode is D8-G1 (docs/tasks/task-kernel-survey-2026-10.md, "KS-D8 design and gates"): mamba_conv_seq,
// mamba_ssm_seq and mamba_gnorm_rows over M rows against M steps of mamba_conv, mamba_ssm and mamba_gnorm, at nemotron-tiny's
// and the 9B's mixer shapes, M = 1, 7 and 64, from a non-zero window and state. The projection rows are f16 values (the row
// kernels read them as f16, decode as f32). Outputs, window and state byte-equal, in the fast-math and the precise library.
// GOINFER_KSD8_PLANT=1 compiles the registered plant (the scan restarting its state at each row) and expects red.
func TestKSD8_seqKernelsMatchDecode(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	src := allKernels
	if os.Getenv("GOINFER_KSD8_PLANT") != "" {
		const at = "            float v = ssm[sBase + n] * dA + dx * cr[bBase + n];"
		if !strings.Contains(src, at) {
			t.Fatal("plant site not found")
		}
		src = strings.Replace(src, at, "            float v = (m == 0u ? ssm[sBase + n] : 0.0f) * dA + dx * cr[bBase + n];", 1)
	}
	type shape struct {
		name                                                        string
		nHeads, headDim, dState, nGroups, dConv, normGroups, dInner int
	}
	shapes := []shape{{"nemotron-tiny", 8, 8, 16, 1, 4, 1, 64}, {"9B", 128, 80, 128, 8, 4, 8, 10240}}
	rng := rand.New(rand.NewSource(61))
	total := 0
	for _, precise := range []bool{false, true} {
		compile := d.CompileLibrary
		if precise {
			compile = d.CompileLibraryPrecise
		}
		lib, err := compile(src, MSL3_1)
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
		pConv, pSSM, pGN := pipe("mamba_conv"), pipe("mamba_ssm"), pipe("mamba_gnorm")
		pConvSeq, pSSMSeq, pGNRows := pipe("mamba_conv_seq"), pipe("mamba_ssm_seq"), pipe("mamba_gnorm_rows")
		q := d.NewCommandQueue()
		for _, sh := range shapes {
			gSize := sh.nGroups * sh.dState
			convDim := sh.dInner + 2*gSize
			projDim := 2*sh.dInner + 2*gSize + sh.nHeads
			K, groupSize := sh.dConv, sh.dInner/sh.normGroups
			u := func(v int) Buffer { return NewBufferU32(d, uint32(v)) }
			f32s := func(n int, sd float64) []float32 {
				v := make([]float32, n)
				for i := range v {
					v[i] = float32(rng.NormFloat64() * sd)
				}
				return v
			}
			convW, convB := NewBufferFloats(d, f32s(convDim*K, 0.3)), NewBufferFloats(d, f32s(convDim, 0.1))
			headP := make([]float32, sh.nHeads*3)
			for h := range sh.nHeads {
				headP[h*3], headP[h*3+1], headP[h*3+2] = -float32(rng.Float64()*2+0.1), float32(rng.NormFloat64()), float32(rng.NormFloat64())
			}
			hp, normW := NewBufferFloats(d, headP), NewBufferFloats(d, f32s(sh.dInner, 1))
			win0, ssm0 := f32s((K-1)*convDim, 0.5), f32s(sh.nHeads*sh.headDim*sh.dState, 0.2)
			uConvDim, uK, uDInner, uProjDim := u(convDim), u(K), u(sh.dInner), u(projDim)
			uNH, uHD, uDS, uRep, uGSize, uDtBase := u(sh.nHeads), u(sh.headDim), u(sh.dState), u(sh.nHeads/sh.nGroups), u(gSize), u(sh.dInner+convDim)
			uGroupSize, uZero, uGroups, uEps := u(groupSize), u(0), u(sh.normGroups), NewBufferFloats(d, []float32{1e-5})
			for _, M := range []int{1, 7, 64} {
				projH := make([]uint16, M*projDim)
				projF := make([]float32, M*projDim)
				for i := range projH {
					projH[i] = f32ToF16(float32(rng.NormFloat64()))
					projF[i] = f16ToF32(projH[i])
				}
				// decode: M steps over one row each
				winD, ssmD := NewBufferFloats(d, slices.Clone(win0)), NewBufferFloats(d, slices.Clone(ssm0))
				row, convR, yR, gR := d.NewBufferLen(projDim), d.NewBufferLen(convDim), d.NewBufferLen(sh.dInner), d.NewBufferLen(sh.dInner)
				var wantConv, wantY, wantG []uint32
				for m := range M {
					copy(row.Floats()[:projDim], projF[m*projDim:(m+1)*projDim])
					e := q.Begin()
					e.Dispatch(pConv, convDim, 64, row, convW, convB, winD, convR, uConvDim, uK, uDInner)
					e.Dispatch(pSSM, sh.nHeads*sh.headDim, 64, convR, row, hp, ssmD, yR, uNH, uHD, uDS, uRep, uGSize, uDInner, uDtBase)
					e.Dispatch(pGN, sh.normGroups*64, 64, yR, row, normW, gR, uGroupSize, uEps, uZero)
					e.End()
					wantConv = append(wantConv, convR.U32s()[:convDim]...)
					wantY = append(wantY, yR.U32s()[:sh.dInner]...)
					wantG = append(wantG, gR.U32s()[:sh.dInner]...)
				}
				// rows: one dispatch per stage over all M
				winS, ssmS := NewBufferFloats(d, slices.Clone(win0)), NewBufferFloats(d, slices.Clone(ssm0))
				projB := NewBufferU16s(d, projH)
				convS, yS, gS := d.NewBufferLen(M*convDim), d.NewBufferLen(M*sh.dInner), d.NewBufferLen(M*sh.dInner)
				uM := u(M)
				e := q.Begin()
				e.Dispatch(pConvSeq, convDim, 64, projB, convW, convB, winS, convS, uConvDim, uK, uDInner, uProjDim, uM)
				e.Dispatch(pSSMSeq, sh.nHeads*sh.headDim, 64, convS, projB, hp, ssmS, yS, uNH, uHD, uDS, uRep, uGSize, uDInner, uDtBase, uConvDim, uProjDim, uM)
				e.Dispatch(pGNRows, M*sh.normGroups*64, 64, yS, projB, normW, gS, uGroupSize, uEps, uZero, uGroups, uDInner, uProjDim)
				e.End()
				diff := func(a, b []uint32) int {
					n := 0
					for i := range a {
						if a[i] != b[i] {
							n++
						}
					}
					return n
				}
				dc, dy, dg := diff(wantConv, convS.U32s()[:M*convDim]), diff(wantY, yS.U32s()[:M*sh.dInner]), diff(wantG, gS.U32s()[:M*sh.dInner])
				dw, ds := diff(winD.U32s()[:len(win0)], winS.U32s()[:len(win0)]), diff(ssmD.U32s()[:len(ssm0)], ssmS.U32s()[:len(ssm0)])
				n := dc + dy + dg + dw + ds
				total += n
				if n != 0 {
					fmt.Fprintf(os.Stderr, "[D8-G1] precise=%v %s M=%d: conv %d, y %d, gated %d, window %d, state %d differ\n", precise, sh.name, M, dc, dy, dg, dw, ds)
					t.Errorf("precise=%v %s M=%d: %d values differ from decode", precise, sh.name, M, n)
				}
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[D8-G1] 12 cells (2 libraries × 2 shapes × M = 1, 7, 64): %d values differ in all\n", total)
}

// TestKSD8_tinyFidelity is D8-G2: nemotron-tiny at int4, prompts of 16, 48 and 100 seeded token ids (8 each), the last
// position's logits from the sequential path (Forward per token), the batched pass (PrefillLast) and a CPU f32 reference. Per
// prompt length, the pass is non-inferior to the sequential path against the reference: its mean KL at most the sequential
// path's + 0.005 nats, and its top-1 agreement at least the sequential path's less one prompt. The pass must be taken.
func TestKSD8_tinyFidelity(t *testing.T) {
	const path = "../testdata/nemotron-tiny"
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: 256})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	if !r.prefillOK || r.mamba == nil {
		t.Fatalf("the batched pass is not admitted (prefillOK %v, mamba %v): %v", r.prefillOK, r.mamba != nil, m.MissingResidentFeatures(prefillFeatures))
	}
	t.Setenv("GOINFER_CPU_FAST_ATTENTION", "0")
	cpu, err := decoder.Load(path, decoder.Options{Backend: "cpu", ResidentContext: 256})
	if err != nil {
		t.Fatalf("cpu reference load: %v", err)
	}
	defer cpu.Close()
	rng := rand.New(rand.NewSource(67))
	for _, M := range []int{16, 48, 100} {
		var klSeq, klPass float64
		topSeq, topPass := 0, 0
		for range 8 {
			ids := make([]int, M)
			for i := range ids {
				ids[i] = rng.Intn(r.V)
			}
			r.resetRecurrent()
			var seq []float32
			for i, id := range ids {
				seq = r.Forward(id, i)
			}
			seq = slices.Clone(seq)
			r.resetRecurrent()
			pass := r.PrefillLast(getEmbs(r, ids), 0)
			if pass == nil {
				t.Fatalf("M=%d: the pass returned no logits: %v", M, r.takeExecErr())
			}
			pass = slices.Clone(pass)
			ref, err := cpu.PrefillLogitsForTest(context.Background(), ids, cpu.NewCache(M))
			if err != nil {
				t.Fatalf("cpu reference: %v", err)
			}
			klSeq += klLogits(ref, seq) / 8
			klPass += klLogits(ref, pass) / 8
			if argmaxF(seq) == argmaxF(ref) {
				topSeq++
			}
			if argmaxF(pass) == argmaxF(ref) {
				topPass++
			}
		}
		ok := klPass <= klSeq+0.005 && topPass >= topSeq-1
		fmt.Fprintf(os.Stderr, "[D8-G2] M=%d: mean KL to the CPU f32 reference, sequential %.4e, pass %.4e; top-1 agreement sequential %d/8, pass %d/8: %v\n",
			M, klSeq, klPass, topSeq, topPass, map[bool]string{true: "PASS", false: "FAIL"}[ok])
		if !ok {
			t.Errorf("M=%d: the pass is inferior to the sequential path against the reference", M)
		}
	}
}

//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"

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

// TestKSD8_scratchWithinPlan is D8-G5's counter half: one batched pass over nemotron-tiny's 100 rows (and, with
// GOINFER_HEAVY_TESTS=1, the 9B's 512) counts at most the bytes prefillScratchBytes plans for it, and the pass ran (it counted
// something). The 9B's load passing the fit guard is the admission half.
func TestKSD8_scratchWithinPlan(t *testing.T) {
	cells := []struct {
		path string
		M    int
	}{{"../testdata/nemotron-tiny", 100}}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		cells = append(cells, struct {
			path string
			M    int
		}{os.ExpandEnv(fmt.Sprintf("$HOME/models/nemotron/nemotron-nano-9b-v2-int4-metal.v%d.giw", decoder.GIWFormatVersion)), 512})
	}
	for _, c := range cells {
		m, err := decoder.Load(c.path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024})
		if err != nil {
			t.Fatalf("load %s: %v", c.path, err)
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok || !a.r.prefillOK {
			m.Close()
			t.Fatalf("%s: the batched pass is not admitted", c.path)
		}
		rng := rand.New(rand.NewSource(71))
		ids := make([]int, c.M)
		for i := range ids {
			ids[i] = rng.Intn(min(a.r.V, 1000))
		}
		a.r.resetRecurrent()
		before := prefillScratchSeen.Load()
		if _, err := a.PrefillLast(context.Background(), getEmbs(a.r, ids), 0); err != nil {
			m.Close()
			t.Fatalf("%s: prefill: %v", c.path, err)
		}
		counted, planned := prefillScratchSeen.Load()-before, a.r.prefillScratchBytes(c.M)
		fmt.Fprintf(os.Stderr, "[D8-G5] %s, %d rows: counted %d bytes, planned %d\n", c.path, c.M, counted, planned)
		if counted <= 0 || counted > planned {
			t.Errorf("%s: counted %d bytes against a plan of %d", c.path, counted, planned)
		}
		m.Close()
	}
}

// ksd8Load9B loads Nemotron-Nano-9B-v2's int4 Metal .giw and builds a resident with the batched pass admitted.
func ksd8Load9B(t *testing.T) (*decoder.Model, *resident) {
	t.Helper()
	path := os.ExpandEnv(fmt.Sprintf("$HOME/models/nemotron/nemotron-nano-9b-v2-int4-metal.v%d.giw", decoder.GIWFormatVersion))
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024})
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok || !a.r.prefillOK || a.r.mamba == nil {
		m.Close()
		t.Fatalf("the batched pass is not admitted on the 9B")
	}
	return m, a.r
}

// ksd8Ids is prompt k's n seeded token ids below the vocabulary's first 32,000.
func ksd8Ids(k, n, V int) []int {
	rng := rand.New(rand.NewSource(int64(1000 + k)))
	ids := make([]int, n)
	for i := range ids {
		ids[i] = rng.Intn(min(V, 32000))
	}
	return ids
}

// TestKSD8_9BFidelity is D8-G3: the 9B, 8 prompts of 512 seeded token ids, the last position's logits from the batched pass
// against the sequential path (no f32 reference fits this box). PASS iff top-1 agrees on at least 7 of 8 and the mean symmetric
// KL ≤ 0.05 nats; 6 of 8 or a KL in (0.05, 0.10]: PARK; worse: FAIL. GOINFER_HEAVY_TESTS=1.
func TestKSD8_9BFidelity(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 9B)")
	}
	m, r := ksd8Load9B(t)
	defer m.Close()
	agree, klSum := 0, 0.0
	for k := range 8 {
		ids := ksd8Ids(k, 512, r.V)
		r.resetRecurrent()
		var seq []float32
		for i, id := range ids {
			seq = r.Forward(id, i)
		}
		seq = slices.Clone(seq)
		r.resetRecurrent()
		pass := r.PrefillLast(getEmbs(r, ids), 0)
		if pass == nil {
			t.Fatalf("prompt %d: the pass returned no logits: %v", k, r.takeExecErr())
		}
		pass = slices.Clone(pass)
		kl := (klLogits(seq, pass) + klLogits(pass, seq)) / 2
		klSum += kl
		if argmaxF(seq) == argmaxF(pass) {
			agree++
		}
		fmt.Fprintf(os.Stderr, "[D8-G3] prompt %d: top-1 sequential %d pass %d, symmetric KL %.4e, cosine %.6f\n", k, argmaxF(seq), argmaxF(pass), kl, cosF(seq, pass))
	}
	mean := klSum / 8
	verdict := "FAIL"
	switch {
	case agree >= 7 && mean <= 0.05:
		verdict = "PASS"
	case agree >= 6 && mean <= 0.10:
		verdict = "PARK"
	}
	fmt.Fprintf(os.Stderr, "[D8-G3] top-1 agrees on %d of 8, mean symmetric KL %.4e: %s\n", agree, mean, verdict)
	if verdict == "FAIL" {
		t.Errorf("the batched pass is not near the sequential path on the 9B")
	}
}

// TestKSD8_9BSpeed is D8-G4: the 9B, a 512-token prompt's time to the first token (all positions through, logits of the
// last), batched against sequential, A B B A over 2 pairs after one warm run of each, each run's start load printed. A timed
// run: hold the timing lock (GOINFER_KSD8_TIMING=1).
func TestKSD8_9BSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSD8_TIMING") == "" {
		t.Skip("set GOINFER_KSD8_TIMING=1 (a timed run under the timing lock)")
	}
	m, r := ksd8Load9B(t)
	defer m.Close()
	ids := ksd8Ids(0, 512, r.V)
	embs := getEmbs(r, ids)
	run := func(batched bool) float64 {
		r.resetRecurrent()
		t0 := time.Now()
		if batched {
			if r.PrefillLast(embs, 0) == nil {
				t.Fatalf("the pass returned no logits: %v", r.takeExecErr())
			}
		} else {
			for i, id := range ids {
				r.Forward(id, i)
			}
		}
		return time.Since(t0).Seconds()
	}
	run(false)
	run(true)
	var speedups []float64
	for i := range 2 {
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		var s, b float64
		if i%2 == 0 {
			s, b = run(false), run(true)
		} else {
			b, s = run(true), run(false)
		}
		speedups = append(speedups, s/b)
		fmt.Fprintf(os.Stderr, "[D8-G4] pair %d (load %s): sequential %.2f s (%.1f tok/s), batched %.2f s (%.1f tok/s), speedup %.2fx\n",
			i, la[:len(la)-1], s, 512/s, b, 512/b, s/b)
	}
	med := (speedups[0] + speedups[1]) / 2
	fmt.Fprintf(os.Stderr, "[D8-G4] median speedup %.2fx: %s\n", med, map[bool]string{true: "SHIP", false: "PARK"}[med >= 2])
}

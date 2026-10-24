//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKSC4_foldMatchesThreeStep is C4-G1 (a) (docs/tasks/task-kernel-survey-2026-10.md, "KS-C4 gates"): the folded pair
// (gemv_w4a8_sa_glu_rows<R>, then gemv_w4a8_resid_staged_hq<R, R>) against gemv_w4a8_sa_rows<R> → swiglu_quant →
// gemv_w4a8_resid_staged<R> at the 0.5B's, 1.5B's and 7B's FFN shapes, R = 2 and 4, SiLU and GELU-tanh, random int4 weights,
// f16 scales and int8 activations, in the fast-math and the precise library. The down outputs (residual added) byte-equal.
func TestKSC4_foldMatchesThreeStep(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	shapes := []struct{ H, I int }{{896, 4864}, {1536, 8960}, {3584, 18944}}
	rng := rand.New(rand.NewSource(41))
	for _, precise := range []bool{false, true} {
		compile := d.CompileLibrary
		if precise {
			compile = d.CompileLibraryPrecise
		}
		lib, err := compile(allKernels, MSL3_1)
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
		q := d.NewCommandQueue()
		for _, s := range shapes {
			H, I := s.H, s.I
			halves := func(n int, sd, mean float64) Buffer {
				v := make([]uint16, n)
				for i := range v {
					v[i] = f32ToF16(float32(rng.NormFloat64()*sd + mean))
				}
				b := d.NewBufferBytes(2 * n)
				copy(b.U16s()[:n], v)
				return b
			}
			words := func(n int) Buffer {
				b := d.NewBufferBytes(4 * n)
				for i, u := 0, b.U32s()[:n]; i < n; i++ {
					u[i] = rng.Uint32()
				}
				return b
			}
			guW, guS := words(2*I*H/8), halves(2*I*H/32, 0.001, 0.002)
			dW, dS := words(H*I/8), halves(H*I/32, 0.001, 0.002)
			aq := d.NewBufferBytes(H)
			for i, v := 0, aq.Int8s()[:H]; i < H; i++ {
				v[i] = int8(rng.Intn(255) - 127)
			}
			init := make([]float32, H)
			for i := range init {
				init[i] = float32(rng.NormFloat64())
			}
			asc, uH, uI := NewBufferFloats(d, []float32{0.0123}), NewBufferU32(d, uint32(H)), NewBufferU32(d, uint32(I))
			for _, act := range []uint32{1, 0} { // ACT_SILU, ACT_GELU_TANH
				uAct := NewBufferU32(d, act)
				for _, R := range []int{2, 4} {
					if (2*I)%(8*R) != 0 || H%(8*R) != 0 {
						continue
					}
					want := NewBufferFloats(d, slices.Clone(init))
					gu, dq, ds := d.NewBufferLen(2*I), d.NewBufferBytes(I), d.NewBufferLen(1)
					e := q.Begin()
					e.DispatchTG(pipe(fmt.Sprintf("gemv_w4a8_sa_rows%d", R)), 2*I*32/R, 256, H*2, guW, guS, aq, asc, gu, uH)
					e.Dispatch(pipe("swiglu_quant"), 256, 256, gu, gu.At(I*4), dq, ds, uI, uAct)
					e.DispatchTG(pipe(fmt.Sprintf("gemv_w4a8_resid_staged%d", R)), H*32/R, 256, I, dW, dS, dq, ds, want, uI)
					e.End()
					got := NewBufferFloats(d, slices.Clone(init))
					h, part := d.NewBufferLen(2*I), d.NewBufferBytes(I)
					e = q.Begin()
					e.DispatchTG(pipe(fmt.Sprintf("gemv_w4a8_sa_glu_rows%d", R)), 2*I*32/R, 256, H*2, guW, guS, aq, asc, h, part, uH, uI, uAct)
					e.DispatchTG(pipe(fmt.Sprintf("gemv_w4a8_resid_staged_hq%d_%d", R, R)), H*32/R, 256, I, dW, dS, h, part, got, uI)
					e.End()
					w, g := want.U32s()[:H], got.U32s()[:H]
					diff := 0
					for i := range w {
						if w[i] != g[i] {
							diff++
						}
					}
					fmt.Fprintf(os.Stderr, "[C4-G1] precise=%v H=%d I=%d act=%d R=%d: %d of %d down outputs differ\n", precise, H, I, act, R, diff, H)
					if diff != 0 {
						t.Errorf("precise=%v H=%d I=%d act=%d R=%d: %d down outputs differ from the three-step form", precise, H, I, act, R, diff)
					}
				}
			}
		}
	}
}

// ksc4Plain reports whether every layer of a's resident takes the folded FFN (encodeLayerResidualWith's conditions).
func ksc4Plain(a *metalResident) bool {
	r := a.r
	if !r.ffnFolded() || r.loraLayers != nil || r.sandwich || r.postOnly || r.parallelBlock {
		return false
	}
	for l := range r.layers {
		if r.canUseF16Lane(l) {
			return false
		}
	}
	return true
}

// TestKSC4_decodeIdentity is C4-G1 (b): llama-tiny (and, with GOINFER_HEAVY_TESTS=1, the 0.5B's and 1.5B's Metal .giw) at
// int4, 32 greedy decode steps, every step's logits with the fold on against off on one resident (pSAGluRows emptied for the
// off arm). Byte-equal. The fold is shown taken by a third arm with the folded down kernel swapped for a no-op, whose logits
// must differ from the fold's.
func TestKSC4_decodeIdentity(t *testing.T) {
	paths := []string{"../testdata/llama-tiny"}
	if os.Getenv("GOINFER_HEAVY_TESTS") != "" {
		for _, n := range []string{"0.5b", "1.5b"} {
			paths = append(paths, os.ExpandEnv(fmt.Sprintf("$HOME/models/qwen2.5-coder-%s-instruct-q4_k_m.int4.metal.v%d.giw", n, decoder.GIWFormatVersion)))
		}
	}
	for _, path := range paths {
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256})
		if err != nil {
			t.Fatalf("load %s: %v", path, err)
		}
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok {
			t.Fatalf("%s: not Metal-resident: %s", path, m.ResidentDecline())
		}
		if !ksc4Plain(a) {
			t.Errorf("%s: the fold does not apply (rows %+v, folded %v)", path, a.r.gemvRows, a.r.ffnFolded())
			m.Close()
			continue
		}
		glu, hq := a.r.pSAGluRows, a.r.pGemvResidHQ
		on := ksc3Decode(t, m, a, 32)
		a.r.pSAGluRows = Pipeline{}
		off := ksc3Decode(t, m, a, 32)
		a.r.pSAGluRows = glu
		lib, err := a.r.d.CompileLibrary("#include <metal_stdlib>\nusing namespace metal;\nkernel void ksc4_noop() {}\n", MSL3_1)
		if err != nil {
			t.Fatal(err)
		}
		noop, err := a.r.d.NewComputePipeline(lib, "ksc4_noop")
		if err != nil {
			t.Fatal(err)
		}
		a.r.pGemvResidHQ = noop
		dead := ksc3Decode(t, m, a, 1)
		a.r.pGemvResidHQ = hq
		d := 0
		for s := range on {
			for i := range on[s] {
				if math.Float32bits(on[s][i]) != math.Float32bits(off[s][i]) {
					d++
				}
			}
		}
		taken := 0
		for i := range dead[0] {
			if math.Float32bits(dead[0][i]) != math.Float32bits(on[0][i]) {
				taken++
			}
		}
		fmt.Fprintf(os.Stderr, "[C4-G1] %s: rows %+v: %d of %d logits differ over 32 steps, fold on against off; the no-op'd fold moves %d of %d\n",
			path, a.r.gemvRows, d, 32*len(on[0]), taken, len(on[0]))
		if d != 0 {
			t.Errorf("%s: %d logits differ between the folded and three-step FFN", path, d)
		}
		if taken == 0 {
			t.Errorf("%s: no-op'ing the folded down kernel changed nothing: the fold is not taken", path)
		}
		m.Close()
	}
}

// TestKSC4_allocatedNotHigher is C4-G3: Metal's allocated size after the 1.5B's resident is built, the fold on against off
// (ksc4FoldOffForTest, read at build). Needs GOINFER_HEAVY_TESTS=1.
func TestKSC4_allocatedNotHigher(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads the 1.5B)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.ExpandEnv(fmt.Sprintf("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v%d.giw", decoder.GIWFormatVersion))
	size := func(off bool) int64 {
		ksc4FoldOffForTest = off
		defer func() { ksc4FoldOffForTest = false }()
		base := d.CurrentAllocatedSize()
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		a, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok || a.r.ffnFolded() == off {
			t.Fatalf("off=%v: resident %v, folded %v", off, ok, ok && a.r.ffnFolded())
		}
		return int64(d.CurrentAllocatedSize()) - int64(base)
	}
	off, on := size(true), size(false)
	fmt.Fprintf(os.Stderr, "[C4-G3] Metal allocated net of the base: fold off %d bytes, on %d bytes\n", off, on)
	if on > off {
		t.Errorf("the fold allocates %d bytes more", on-off)
	}
}

// TestKSC4_decodeSpeed is C4-G2: the 1.5B's Metal .giw at int4, 128 greedy decode tokens after a 32-token prompt (Forward
// steps), the fold on against off on one resident, A B B A over 10 pairs after 2 warm runs each, each pair's start load
// printed. A timed run: hold the timing lock (GOINFER_KSC4_TIMING=1). A fit refusal prints FIT-REFUSED.
func TestKSC4_decodeSpeed(t *testing.T) {
	if os.Getenv("GOINFER_KSC4_TIMING") == "" {
		t.Skip("set GOINFER_KSC4_TIMING=1 (a timed run under the timing lock)")
	}
	path := os.ExpandEnv(fmt.Sprintf("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.v%d.giw", decoder.GIWFormatVersion))
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 256})
	if err != nil {
		if strings.Contains(err.Error(), "NOT loaded") {
			fmt.Fprintf(os.Stderr, "[C4-G2] FIT-REFUSED: %v\n", err)
		}
		t.Fatal(err)
	}
	defer m.Close()
	a, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("not Metal-resident: %s", m.ResidentDecline())
	}
	if !ksc4Plain(a) {
		t.Fatalf("the fold does not apply (rows %+v)", a.r.gemvRows)
	}
	glu := a.r.pSAGluRows
	const prompt, gen = 32, 128
	run := func(fold bool) float64 {
		if !fold {
			a.r.pSAGluRows = Pipeline{}
		}
		defer func() { a.r.pSAGluRows = glu }()
		a.Reset()
		tok := 13
		for pos := range prompt {
			if err := a.ForwardNoLogits(m.EmbedResidentForTest((pos*7919+13)%1000), pos); err != nil {
				t.Fatal(err)
			}
		}
		t0 := time.Now()
		for pos := prompt; pos < prompt+gen; pos++ {
			lg, err := a.Forward(m.EmbedResidentForTest(tok), pos)
			if err != nil {
				t.Fatal(err)
			}
			tok = 0
			for i, v := range lg {
				if v > lg[tok] {
					tok = i
				}
			}
		}
		return time.Since(t0).Seconds() / gen
	}
	for range 2 {
		run(false)
		run(true)
	}
	var ratios []float64
	below := 0
	for i := range 10 {
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		var off, on float64
		if i%2 == 0 {
			off, on = run(false), run(true)
		} else {
			on, off = run(true), run(false)
		}
		ratios = append(ratios, on/off)
		if on < off {
			below++
		}
		fmt.Fprintf(os.Stderr, "[C4-G2] pair %d (load %s): off %.3f ms/token, on %.3f ms/token, on/off %.4f\n", i, la[:len(la)-1], off*1e3, on*1e3, on/off)
	}
	slices.Sort(ratios)
	fmt.Fprintf(os.Stderr, "[C4-G2] median on/off %.4f, %d of 10 pairs below 1 (rows %+v)\n", (ratios[4]+ratios[5])/2, below, a.r.gemvRows)
}

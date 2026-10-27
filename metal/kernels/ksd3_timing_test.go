//go:build darwin

package kernels

import (
	"fmt"
	"math/rand"
	"os"
	"slices"
	"testing"
)

// TestKSD3_scanKernelTiming is D3-G3 (docs/tasks/task-kernel-survey-2026-10.md, "KS-D3 gates"): the GPU time (GPUEnd −
// GPUStart, one command buffer per dispatch) of delta_rule_seq against delta_rule_seq_tg alone at M = 512, Qwen3.5 geometry,
// interleaved A B B A over 10 pairs after 3 warm runs of each; delta_conv_seq against delta_conv_seq_reg the same way, read
// only. A timed run: hold the timing lock.
//
//	GOINFER_KSD3_TIMING=1 python3 scripts/timing_lock.py run --label ksd3-g3 -- go test -count=1 -run '^TestKSD3_scanKernelTiming$' -v ./metal/
func TestKSD3_scanKernelTiming(t *testing.T) {
	if os.Getenv("GOINFER_KSD3_TIMING") == "" {
		t.Skip("set GOINFER_KSD3_TIMING=1 (a timed run, by day, under the timing lock)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(name string) Pipeline {
		p, e := d.NewComputePipeline(lib, name)
		if e != nil {
			t.Fatalf("pipeline %s: %v", name, e)
		}
		return p
	}
	const (
		hk, hv, nk, nv, convK, M = 128, 128, 16, 48, 4, 512
		rep, keyDim, valueDim    = nv / nk, nk * hk, nv * hv
		convDim                  = 2*keyDim + valueDim
	)
	rng := rand.New(rand.NewSource(3))
	rnd := func(n int, scale float64) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * scale)
		}
		return s
	}
	u := func(v int) Buffer { return NewBufferU32(d, uint32(v)) }
	headP := rnd(M*nv*2, 0.3)
	for i := range headP {
		headP[i] = 0.5 + headP[i]*0.1 // beta and the decay gate near (0, 1)
	}
	qn, kn, conv, state0 := NewBufferFloats(d, rnd(M*keyDim, 0.1)), NewBufferFloats(d, rnd(M*keyDim, 0.1)), NewBufferFloats(d, rnd(M*convDim, 1)), rnd(nv*hv*hk, 0.05)
	dHead, state, core := NewBufferFloats(d, headP), NewBufferFloats(d, state0), d.NewBufferLen(M*valueDim)
	mixed := make([]uint16, M*convDim)
	for i := range mixed {
		mixed[i] = f32ToF16(float32(rng.NormFloat64()))
	}
	dMixed, convW, win0 := NewBufferU16s(d, mixed), NewBufferFloats(d, rnd(convDim*convK, 0.5)), rnd((convK-1)*convDim, 0.5)
	win, convOut := NewBufferFloats(d, win0), d.NewBufferLen(M*convDim)
	q := d.NewCommandQueue()
	uNv, uHk, uHv, uRep, uVBase, uKeyDim, uConvDim, uM, uK := u(nv), u(hk), u(hv), u(rep), u(2*keyDim), u(keyDim), u(convDim), u(M), u(convK)

	timeRule := func(p Pipeline, tg int) float64 {
		copy(state.Floats(), state0)
		e := q.Begin()
		e.Dispatch(p, valueDim, tg, qn, kn, conv, dHead, state, core, uNv, uHk, uHv, uRep, uVBase, uKeyDim, uConvDim, uM)
		e.End()
		return e.GPUEnd() - e.GPUStart()
	}
	timeConv := func(p Pipeline) float64 {
		copy(win.Floats(), win0)
		e := q.Begin()
		e.Dispatch(p, convDim, 256, dMixed, convW, win, convOut, uConvDim, uK, uM)
		e.End()
		return e.GPUEnd() - e.GPUStart()
	}
	ab := func(name string, old, cur func() float64) (median float64, below int) {
		for range 3 {
			old()
			cur()
		}
		var ratios []float64
		for i := range 10 {
			var a, b float64
			if i%2 == 0 {
				a, b = old(), cur()
			} else {
				b, a = cur(), old()
			}
			ratios = append(ratios, b/a)
			if b < a {
				below++
			}
			fmt.Fprintf(os.Stderr, "[D3-G3] %s pair %d: old %.3f ms, new %.3f ms, new/old %.4f\n", name, i, a*1e3, b*1e3, b/a)
		}
		slices.Sort(ratios)
		median = (ratios[4] + ratios[5]) / 2
		fmt.Fprintf(os.Stderr, "[D3-G3] %s: median new/old %.4f, %d of 10 pairs below 1\n", name, median, below)
		return median, below
	}
	oldRule, newRule := pipe("delta_rule_seq"), pipe("delta_rule_seq_tg")
	ab("delta_rule_seq", func() float64 { return timeRule(oldRule, 128) }, func() float64 { return timeRule(newRule, 32) })
	oldConv, newConv := pipe("delta_conv_seq"), pipe("delta_conv_seq_reg")
	ab("delta_conv_seq (read only)", func() float64 { return timeConv(oldConv) }, func() float64 { return timeConv(newConv) })
}

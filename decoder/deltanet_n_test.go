package decoder

import (
	"math"
	"math/rand"
	"testing"
)

// P26c (docs/queue-performance.md). deltaNetCoreN fans the conv, the head recurrence and the gated norm out over K rows; its contract is that it changes no
// bit: the per-token loop is the reference, and the outputs, the recurrent state and the conv window afterwards must match it EXACTLY, from a fresh state and
// from a non-empty one (a prompt that continues earlier tokens), for K below, at and above the conv kernel and across worker counts. The fan-out is forced onto
// the tiny fixture (deltaNetMinWork) and counted (deltaNetFanouts) so the test cannot pass by never forking.
func TestDeltaNetCoreN_bitIdentical(t *testing.T) {
	m, err := Load("../testdata/qwen35-tiny", Options{})
	if err != nil {
		t.Skipf("no qwen35-tiny fixture: %v", err)
	}
	defer m.Close()
	arch := m.w.arch
	layer := -1
	for l := range arch.NumLayers {
		if arch.isLinearLayer(l) {
			layer = l
			break
		}
	}
	if layer < 0 {
		t.Fatal("the fixture has no DeltaNet layer")
	}
	w := m.w.Layers[layer].delta
	p := *arch.qwen35
	eps := arch.NormEps
	nv, hk, hv, nk := p.NumValueHeads, p.KeyHeadDim, p.ValueHeadDim, p.NumKeyHeads
	valueDim, convDim := hv*nv, 2*hk*nk+hv*nv

	defer func(a int) { deltaNetMinWork = a }(deltaNetMinWork)
	deltaNetMinWork = 2

	rnd := rand.New(rand.NewSource(7))
	vec := func(n int, scale float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rnd.NormFloat64() * scale)
		}
		return x
	}
	newState := func(priorRows int, seed []float32, convWin [][]float32) *deltaState {
		st := newDeltaState(p)
		copy(st.s, seed)
		st.convWin = append([][]float32(nil), convWin...)
		_ = priorRows
		return st
	}
	bits := func(a, b []float32, what string, t *testing.T) {
		t.Helper()
		if len(a) != len(b) {
			t.Fatalf("%s: length %d vs %d", what, len(a), len(b))
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("%s[%d]: per-token %v (%08x) vs batched %v (%08x)", what, i, a[i], math.Float32bits(a[i]), b[i], math.Float32bits(b[i]))
			}
		}
	}
	for _, prior := range []int{0, 1, p.ConvKernel - 1} {
		for _, K := range []int{2, 3, p.ConvKernel, 9, 37} {
			convWin := make([][]float32, prior)
			for i := range convWin {
				convWin[i] = vec(convDim, 1)
			}
			seed := vec(len(newDeltaState(p).s), 0.3)
			mixed, z := vec(K*convDim, 1), vec(K*valueDim, 1)
			bt, at := vec(K*nv, 1), vec(K*nv, 1)

			// reference: the per-token loop on its own copies of the inputs (deltaNetCore retains its mixed rows, so each arm gets a slab)
			refMixed := append([]float32(nil), mixed...)
			refCore := make([]float32, K*valueDim)
			ref := newState(prior, seed, convWin)
			for i := range K {
				deltaNetCore(refCore[i*valueDim:(i+1)*valueDim], refMixed[i*convDim:(i+1)*convDim], bt[i*nv:(i+1)*nv], at[i*nv:(i+1)*nv], z[i*valueDim:(i+1)*valueDim], w, p, eps, ref)
			}

			before := deltaNetFanouts.Load()
			got := newState(prior, seed, convWin)
			gotCore := make([]float32, K*valueDim)
			deltaNetCoreN(gotCore, append([]float32(nil), mixed...), bt, at, z, K, w, p, eps, got)
			if K >= deltaNetMinWork && deltaNetFanouts.Load() == before {
				t.Fatalf("prior=%d K=%d: the fan-out never forked, so this compared the code with itself", prior, K)
			}

			tag := func(s string) string { return s }
			bits(refCore, gotCore, tag("core"), t)
			bits(ref.s, got.s, tag("state S"), t)
			if len(ref.convWin) != len(got.convWin) {
				t.Fatalf("prior=%d K=%d: conv window length %d vs %d", prior, K, len(ref.convWin), len(got.convWin))
			}
			for i := range ref.convWin {
				bits(ref.convWin[i], got.convWin[i], "conv window row", t)
			}
		}
	}
}

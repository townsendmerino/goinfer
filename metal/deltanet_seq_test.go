//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// TestDeltaNetSeqKernels_matchDecodeBitwise is D-B01's kernel gate (docs/tasks/task-metal-audit-2026-10.md): the mixer's
// row kernels (delta_conv_seq, delta_gates_rows, delta_norm_rows, delta_rule_seq, delta_gnorm_rows), each dispatched
// once over M rows, against the decode kernels they copy run M times, one row at a time, on the same inputs. Every
// row's output of every stage, the conv window and the recurrent state after the last row must be equal bit for bit.
// The window and state start non-zero, as a continuation from an earlier prompt leaves them. Inputs that arrive as
// half on the batched lane (the projection GEMMs' outputs: mixed and z) are given to both sides as the same halves.
// Real Qwen3.5 geometry (nk = 16, nv = 48, hk = hv = 128, K = 4), and M = 1, 7 and 33.
func TestDeltaNetSeqKernels_matchDecodeBitwise(t *testing.T) {
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
	pConv, pGates, pNorm, pRule, pGNorm := pipe("delta_conv"), pipe("delta_gates"), pipe("delta_norm"), pipe("delta_rule"), pipe("delta_gnorm")
	sConv, sGates, sNorm, sRule, sGNorm := pipe("delta_conv_seq"), pipe("delta_gates_rows"), pipe("delta_norm_rows"), pipe("delta_rule_seq"), pipe("delta_gnorm_rows")

	const (
		hk, hv   = 128, 128
		nk, nv   = 16, 48
		rep      = nv / nk
		convK    = 4
		keyDim   = nk * hk
		valueDim = nv * hv
		convDim  = 2*keyDim + valueDim
		eps      = 1e-6
	)
	rng := rand.New(rand.NewSource(7))
	rnd := func(n int, scale float64) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * scale)
		}
		return s
	}
	halves := func(n int, scale float64) []uint16 {
		h := make([]uint16, n)
		for i := range h {
			h[i] = f32ToF16(float32(rng.NormFloat64() * scale))
		}
		return h
	}
	widen := func(h []uint16) []float32 {
		f := make([]float32, len(h))
		for i, x := range h {
			f[i] = f16ToF32(x)
		}
		return f
	}
	convW, dtBias, normW := rnd(convDim*convK, 0.5), rnd(nv, 0.1), rnd(hv, 0.1)
	negExpA := rnd(nv, 0.5)
	for i, a := range negExpA {
		negExpA[i] = float32(-math.Exp(float64(a))) // -exp(A_log), as the loader precomputes it
	}
	win0, state0 := rnd((convK-1)*convDim, 0.5), rnd(nv*hv*hk, 0.05)
	u := func(v int) Buffer { return NewBufferU32(d, uint32(v)) }
	uConvDim, uK, uNv, uNk, uHk, uHv := u(convDim), u(convK), u(nv), u(nk), u(hk), u(hv)
	uKeyDim, uRep, uVBase := u(keyDim), u(rep), u(2*keyDim)
	uQScale, uEps := NewBufferFloats(d, []float32{float32(1 / math.Sqrt(hk))}), NewBufferFloats(d, []float32{eps})
	dConvW, dDt, dNegA, dNormW := NewBufferFloats(d, convW), NewBufferFloats(d, dtBias), NewBufferFloats(d, negExpA), NewBufferFloats(d, normW)
	q := d.NewCommandQueue()

	for _, M := range []int{1, 7, 33} {
		mixedH, zH := halves(M*convDim, 1.0), halves(M*valueDim, 1.0)
		bt, at := rnd(M*nv, 1.0), rnd(M*nv, 1.0)
		mixed, z := widen(mixedH), widen(zH)

		// decode: one row at a time
		dMixed, dConv, dBt, dAt := d.NewBufferLen(convDim), d.NewBufferLen(convDim), d.NewBufferLen(nv), d.NewBufferLen(nv)
		dHeadP, dQn, dKn := d.NewBufferLen(nv*2), d.NewBufferLen(keyDim), d.NewBufferLen(keyDim)
		dCore, dZ, dOut := d.NewBufferLen(valueDim), d.NewBufferLen(valueDim), d.NewBufferLen(valueDim)
		dWin, dState := NewBufferFloats(d, append([]float32(nil), win0...)), NewBufferFloats(d, append([]float32(nil), state0...))
		var wConv, wHeadP, wQn, wKn, wCore, wOut []float32
		for r := range M {
			copy(dMixed.Floats(), mixed[r*convDim:(r+1)*convDim])
			copy(dBt.Floats(), bt[r*nv:(r+1)*nv])
			copy(dAt.Floats(), at[r*nv:(r+1)*nv])
			copy(dZ.Floats(), z[r*valueDim:(r+1)*valueDim])
			e := q.Begin()
			e.Dispatch(pConv, convDim, 256, dMixed, dConvW, dWin, dConv, uConvDim, uK)
			e.Dispatch(pGates, nv, 64, dBt, dAt, dDt, dNegA, dHeadP, uNv)
			e.Dispatch(pNorm, nk*128, 128, dConv, dQn, dKn, uNk, uHk, uKeyDim, uQScale)
			e.Dispatch(pRule, nv*hv, 128, dQn, dKn, dConv, dHeadP, dState, dCore, uNv, uHk, uHv, uRep, uVBase)
			e.Dispatch(pGNorm, nv*128, 128, dCore, dZ, dNormW, dOut, uNv, uHv, uEps)
			e.End()
			wConv = append(wConv, dConv.Floats()[:convDim]...)
			wHeadP = append(wHeadP, dHeadP.Floats()[:2*nv]...)
			wQn = append(wQn, dQn.Floats()[:keyDim]...)
			wKn = append(wKn, dKn.Floats()[:keyDim]...)
			wCore = append(wCore, dCore.Floats()[:valueDim]...)
			wOut = append(wOut, dOut.Floats()[:valueDim]...)
		}

		// batched: one dispatch per stage over all M rows
		sMixed, sZ := NewBufferU16s(d, mixedH), NewBufferU16s(d, zH)
		sBt, sAt := NewBufferFloats(d, bt), NewBufferFloats(d, at)
		sConvOut, sHeadP := d.NewBufferLen(M*convDim), d.NewBufferLen(M*nv*2)
		sQn, sKn, sCore, sOut := d.NewBufferLen(M*keyDim), d.NewBufferLen(M*keyDim), d.NewBufferLen(M*valueDim), d.NewBufferLen(M*valueDim)
		sWin, sState := NewBufferFloats(d, append([]float32(nil), win0...)), NewBufferFloats(d, append([]float32(nil), state0...))
		uM := u(M)
		e := q.Begin()
		e.Dispatch(sConv, convDim, 256, sMixed, dConvW, sWin, sConvOut, uConvDim, uK, uM)
		e.Dispatch(sGates, M*nv, 64, sBt, sAt, dDt, dNegA, sHeadP, uNv, uM)
		e.Dispatch(sNorm, M*nk*128, 128, sConvOut, sQn, sKn, uNk, uHk, uKeyDim, uQScale, uConvDim, uM)
		e.Dispatch(sRule, nv*hv, 128, sQn, sKn, sConvOut, sHeadP, sState, sCore, uNv, uHk, uHv, uRep, uVBase, uKeyDim, uConvDim, uM)
		e.Dispatch(sGNorm, M*nv*128, 128, sCore, sZ, dNormW, sOut, uNv, uHv, uEps, uM)
		e.End()

		for _, c := range []struct {
			name      string
			want, got []float32
		}{
			{"conv", wConv, sConvOut.Floats()[:M*convDim]},
			{"gates", wHeadP, sHeadP.Floats()[:M*nv*2]},
			{"norm q", wQn, sQn.Floats()[:M*keyDim]},
			{"norm k", wKn, sKn.Floats()[:M*keyDim]},
			{"rule", wCore, sCore.Floats()[:M*valueDim]},
			{"gnorm", wOut, sOut.Floats()[:M*valueDim]},
			{"window after", dWin.Floats()[:(convK-1)*convDim], sWin.Floats()[:(convK-1)*convDim]},
			{"state after", dState.Floats()[:nv*hv*hk], sState.Floats()[:nv*hv*hk]},
		} {
			differ, first := 0, -1
			for i := range c.want {
				if math.Float32bits(c.want[i]) != math.Float32bits(c.got[i]) {
					if first < 0 {
						first = i
					}
					differ++
				}
			}
			if differ != 0 {
				t.Errorf("M=%d %s: %d of %d values differ from decode's, first at %d (%g vs %g)", M, c.name, differ, len(c.want), first, c.want[first], c.got[first])
			}
		}
		t.Logf("M=%d: every stage, the window and the state equal decode's run %d times", M, M)
	}
}

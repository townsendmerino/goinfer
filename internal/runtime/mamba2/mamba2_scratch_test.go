package mamba2

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/internal/runtime/hybridattn"
)

// mixAllocating is Mix as it stood with a fresh conv, y and gated slice on every call: the arm the scratch-reusing Mix is
// held bit-identical to. Keep it frozen; it is the reference, not a second implementation to maintain.
func mixAllocating(proj []float32, w Weights, p Params, eps float64, st *State, f32 bool) []float32 {
	dInner, convDim, K := p.DInner(), p.ConvDim(), p.DConv
	N, P := p.DState, p.HeadDim
	gSize := p.NGroups * N
	repeat := p.NHeads / p.NGroups
	z := proj[:dInner]
	xBC := proj[dInner : dInner+convDim]
	dt := proj[dInner+convDim:]
	conv := make([]float32, convDim)
	win := st.ConvWin
	for c := range convDim {
		s := w.ConvB[c] + w.ConvW[c*K+(K-1)]*xBC[c]
		for j := 0; j < K-1; j++ {
			if idx := len(win) - (K - 1) + j; idx >= 0 {
				s += w.ConvW[c*K+j] * win[idx][c]
			}
		}
		conv[c] = hybridattn.Silu(s)
	}
	switch {
	case len(st.ConvWin) < K-1:
		st.ConvWin = append(st.ConvWin, append(make([]float32, 0, convDim), xBC...))
	case K > 1:
		oldest := st.ConvWin[0]
		copy(st.ConvWin, st.ConvWin[1:])
		st.ConvWin[K-2] = oldest
		copy(oldest, xBC)
	}
	x := conv[:dInner]
	B := conv[dInner : dInner+gSize]
	C := conv[dInner+gSize : dInner+2*gSize]
	y := make([]float32, dInner)
	for head := range p.NHeads {
		g := head / repeat
		Bh := B[g*N : (g+1)*N]
		Ch := C[g*N : (g+1)*N]
		xh := x[head*P : (head+1)*P]
		A := -math.Exp(float64(w.ALog[head]))
		dth := hybridattn.Softplusf(dt[head] + w.DtBias[head])
		var dA float32
		if f32 {
			dA = float32(math.Exp(float64(float32(dth) * float32(A))))
		} else {
			dA = float32(math.Exp(float64(dth) * A))
		}
		S := st.SSM[head*P*N : (head+1)*P*N]
		out := y[head*P : (head+1)*P]
		for pi := range P {
			var acc float32
			row := S[pi*N : pi*N+N]
			dx := dth * xh[pi]
			n := 0
			for ; n+3 < N; n += 4 {
				row[n] = row[n]*dA + dx*Bh[n]
				acc += row[n] * Ch[n]
				row[n+1] = row[n+1]*dA + dx*Bh[n+1]
				acc += row[n+1] * Ch[n+1]
				row[n+2] = row[n+2]*dA + dx*Bh[n+2]
				acc += row[n+2] * Ch[n+2]
				row[n+3] = row[n+3]*dA + dx*Bh[n+3]
				acc += row[n+3] * Ch[n+3]
			}
			for ; n < N; n++ {
				row[n] = row[n]*dA + dx*Bh[n]
				acc += row[n] * Ch[n]
			}
			out[pi] = acc + w.D[head]*xh[pi]
		}
	}
	gated := make([]float32, dInner)
	for i := range dInner {
		gated[i] = y[i] * hybridattn.Silu(z[i])
	}
	GatedRMSNormGrouped(gated, w.NormW, p.NormGroups, eps, f32)
	return gated
}

func scratchTestWeights(p Params, rng *rand.Rand) Weights {
	vec := func(n int, scale float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((rng.Float64()*2 - 1) * scale)
		}
		return v
	}
	return Weights{ConvW: vec(p.ConvDim()*p.DConv, 0.5), ConvB: vec(p.ConvDim(), 0.1), ALog: vec(p.NHeads, 1),
		D: vec(p.NHeads, 1), DtBias: vec(p.NHeads, 0.5), NormW: vec(p.DInner(), 1)}
}

// TestMix_bitIdenticalToTheAllocatingMixer: Mix, reusing its state's scratch from step to step, returns the same bits and
// leaves the same window and SSM state as the frozen allocating mixer on the same inputs, in both precision paths, on shapes
// that reach the unrolled scan and its tail, grouped B and C, a grouped norm, and a one-tap conv. The returned slice is
// compared before the next step, and the next step must not read it as anything but scratch.
func TestMix_bitIdenticalToTheAllocatingMixer(t *testing.T) {
	const eps = 1e-5
	for _, p := range []Params{
		{NHeads: 2, HeadDim: 4, DState: 8, NGroups: 1, DConv: 4, NormGroups: 1},
		{NHeads: 4, HeadDim: 3, DState: 5, NGroups: 2, DConv: 3, NormGroups: 2},
		{NHeads: 8, HeadDim: 16, DState: 32, NGroups: 2, DConv: 4, NormGroups: 2},
		{NHeads: 2, HeadDim: 2, DState: 7, NGroups: 2, DConv: 1, NormGroups: 0},
	} {
		for _, f32 := range []bool{false, true} {
			t.Run(fmt.Sprintf("h%d_p%d_n%d_g%d_k%d_norm%d_f32%v", p.NHeads, p.HeadDim, p.DState, p.NGroups, p.DConv, p.NormGroups, f32), func(t *testing.T) {
				rng := rand.New(rand.NewPCG(11, uint64(p.DState*10+p.DConv)))
				w := scratchTestWeights(p, rng)
				got, want := NewState(p), NewState(p)
				for step := range 10 {
					proj := make([]float32, p.DInner()+p.ConvDim()+p.NHeads)
					for i := range proj {
						proj[i] = float32(rng.Float64()*2 - 1)
					}
					wantOut := mixAllocating(slices.Clone(proj), w, p, eps, want, f32)
					gotOut := Mix(slices.Clone(proj), w, p, eps, got, f32)
					bitsEqual(t, fmt.Sprintf("step %d output", step), gotOut, wantOut)
					bitsEqual(t, fmt.Sprintf("step %d SSM state", step), got.SSM, want.SSM)
					if len(got.ConvWin) != len(want.ConvWin) {
						t.Fatalf("step %d: window holds %d rows, the allocating mixer %d", step, len(got.ConvWin), len(want.ConvWin))
					}
					for i := range want.ConvWin {
						bitsEqual(t, fmt.Sprintf("step %d window row %d", step, i), got.ConvWin[i], want.ConvWin[i])
					}
				}
			})
		}
	}
}

// TestMix_noAllocsOnceWarm: once the conv window is full, a Mix step allocates nothing; its conv and output buffers are the
// state's grow-only scratch.
func TestMix_noAllocsOnceWarm(t *testing.T) {
	p := Params{NHeads: 64, HeadDim: 64, DState: 128, NGroups: 8, DConv: 4, NormGroups: 8}
	rng := rand.New(rand.NewPCG(13, 17))
	w := scratchTestWeights(p, rng)
	st := NewState(p)
	proj := make([]float32, p.DInner()+p.ConvDim()+p.NHeads)
	for i := range proj {
		proj[i] = float32(rng.Float64()*2-1) * 0.1
	}
	for range p.DConv {
		Mix(proj, w, p, 1e-5, st, false)
	}
	if allocs := testing.AllocsPerRun(20, func() { Mix(proj, w, p, 1e-5, st, false) }); allocs != 0 {
		t.Fatalf("a warm Mix step allocates %v times, want 0", allocs)
	}
}

func bitsEqual(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] = %g (%#x), want %g (%#x)", what, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

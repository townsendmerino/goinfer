package mamba2

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// refMixer is one Mamba-2 mixer step written out plainly in float64 from the math rather than from Mix: the depthwise causal
// conv over xBC (window oldest first, zero before the first token) plus bias and SiLU; per head, dt = softplus(dt+bias), the
// state decays by exp(dt·A) with A = −exp(ALog) and gains dt·x⊗B, and y = S·C + D·x; then y·SiLU(z), RMS-normalized per norm
// group, times the weight.
type refMixer struct {
	p   Params
	w   Weights
	win [][]float64
	ssm []float64
}

func silu64(x float64) float64 { return x / (1 + math.Exp(-x)) }

func (r *refMixer) step(proj []float64, eps float64) []float64 {
	p, w := r.p, r.w
	dInner, convDim, K, N, P := p.DInner(), p.ConvDim(), p.DConv, p.DState, p.HeadDim
	z, xBC, dt := proj[:dInner], proj[dInner:dInner+convDim], proj[dInner+convDim:]
	conv := make([]float64, convDim)
	for c := range convDim {
		s := float64(w.ConvB[c])
		for j := range K {
			tap := xBC[c] // the last tap is this token; each earlier tap is an earlier token, or zero before the first
			if back := K - 1 - j; back > 0 {
				tap = 0
				if i := len(r.win) - back; i >= 0 {
					tap = r.win[i][c]
				}
			}
			s += float64(w.ConvW[c*K+j]) * tap
		}
		conv[c] = silu64(s)
	}
	r.win = append(r.win, append([]float64(nil), xBC...))
	gSize := p.NGroups * N
	x, B, C := conv[:dInner], conv[dInner:dInner+gSize], conv[dInner+gSize:dInner+2*gSize]
	y := make([]float64, dInner)
	for h := range p.NHeads {
		g := h / (p.NHeads / p.NGroups)
		A := -math.Exp(float64(w.ALog[h]))
		d := math.Log1p(math.Exp(dt[h] + float64(w.DtBias[h])))
		for pi := range P {
			var acc float64
			for n := range N {
				i := (h*P+pi)*N + n
				r.ssm[i] = r.ssm[i]*math.Exp(d*A) + d*x[h*P+pi]*B[g*N+n]
				acc += r.ssm[i] * C[g*N+n]
			}
			y[h*P+pi] = acc + float64(w.D[h])*x[h*P+pi]
		}
	}
	groups := max(p.NormGroups, 1)
	size := dInner / groups
	out := make([]float64, dInner)
	for g := range groups {
		var ss float64
		for i := g * size; i < (g+1)*size; i++ {
			v := y[i] * silu64(z[i])
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss/float64(size)+eps)
		for i := g * size; i < (g+1)*size; i++ {
			out[i] = float64(w.NormW[i]) * y[i] * silu64(z[i]) * inv
		}
	}
	return out
}

// Mix matches the plain mixer step over several tokens, in its f64 path (production) and its f32 path (the GPU kernels'
// precision, a localization seam) within f32 rounding, on shapes that reach every path: a state size that is a multiple of 4
// (the unrolled scan) and one that is not (its scalar tail), B and C shared by groups of heads, the norm over the whole width
// and per group, and NormGroups 0, which is one group.
func TestMix_matchesThePlainMixerStep(t *testing.T) {
	const eps = 1e-5
	for _, p := range []Params{
		{NHeads: 2, HeadDim: 4, DState: 8, NGroups: 1, DConv: 4, NormGroups: 1},
		{NHeads: 4, HeadDim: 3, DState: 5, NGroups: 2, DConv: 3, NormGroups: 2},
		{NHeads: 2, HeadDim: 2, DState: 7, NGroups: 2, DConv: 2, NormGroups: 0},
	} {
		for _, f32 := range []bool{false, true} {
			t.Run(fmt.Sprintf("h%d_p%d_n%d_g%d_k%d_norm%d_f32%v", p.NHeads, p.HeadDim, p.DState, p.NGroups, p.DConv, p.NormGroups, f32), func(t *testing.T) {
				rng := rand.New(rand.NewPCG(7, uint64(p.DState)))
				vec := func(n int, scale float64) []float32 {
					v := make([]float32, n)
					for i := range v {
						v[i] = float32((rng.Float64()*2 - 1) * scale)
					}
					return v
				}
				w := Weights{ConvW: vec(p.ConvDim()*p.DConv, 0.5), ConvB: vec(p.ConvDim(), 0.1), ALog: vec(p.NHeads, 1),
					D: vec(p.NHeads, 1), DtBias: vec(p.NHeads, 0.5), NormW: vec(p.DInner(), 1)}
				st := NewState(p)
				ref := &refMixer{p: p, w: w, ssm: make([]float64, len(st.SSM))}
				tol := 1e-5
				if f32 {
					tol = 1e-4
				}
				for step := range 4 {
					proj := vec(p.DInner()+p.ConvDim()+p.NHeads, 1)
					proj64 := make([]float64, len(proj))
					for i, v := range proj {
						proj64[i] = float64(v)
					}
					got := Mix(proj, w, p, eps, st, f32)
					want := ref.step(proj64, eps)
					for i := range got {
						if d := math.Abs(float64(got[i]) - want[i]); d > tol*(1+math.Abs(want[i])) {
							t.Fatalf("step %d: out[%d] = %g, the plain mixer %g", step, i, got[i], want[i])
						}
					}
					for i := range st.SSM {
						if d := math.Abs(float64(st.SSM[i]) - ref.ssm[i]); d > tol*(1+math.Abs(ref.ssm[i])) {
							t.Fatalf("step %d: state[%d] = %g, the plain mixer %g", step, i, st.SSM[i], ref.ssm[i])
						}
					}
				}
			})
		}
	}
}

package hybridattn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// refRecurrence is the gated delta rule for one token, written out plainly in float64 from the math rather than from
// DeltaNetRecurrence: per value head, q and k are its key head's L2-normalized query (scaled by 1/sqrt(hk)) and key, the state
// decays by exp(negExpA·softplus(a+dtBias)), the write is beta·(v − Sᵀk) with beta = sigmoid(b) (twice that with NegEigval), S
// gains k ⊗ write, and the output is Sᵀq.
func refRecurrence(conv, at, bt, dtBias, negExpA []float64, p DeltaNetParams, s []float64) []float64 {
	nk, nv, hk, hv := p.NumKeyHeads, p.NumValueHeads, p.KeyHeadDim, p.ValueHeadDim
	keyDim, rep := hk*nk, nv/nk
	norm := func(x []float64, scale float64) []float64 {
		var ss float64
		for _, v := range x {
			ss += v * v
		}
		out := make([]float64, len(x))
		for i, v := range x {
			out[i] = v / math.Sqrt(ss+1e-6) * scale
		}
		return out
	}
	out := make([]float64, nv*hv)
	for h := range nv {
		kh := h / rep
		q := norm(conv[kh*hk:(kh+1)*hk], 1/math.Sqrt(float64(hk)))
		k := norm(conv[keyDim+kh*hk:keyDim+(kh+1)*hk], 1)
		v := conv[2*keyDim+h*hv : 2*keyDim+(h+1)*hv]
		decay := math.Exp(negExpA[h] * math.Log1p(math.Exp(at[h]+dtBias[h])))
		beta := 1 / (1 + math.Exp(-bt[h]))
		if p.NegEigval {
			beta *= 2
		}
		S := s[h*hk*hv : (h+1)*hk*hv] // [hk][hv]
		for i := range S {
			S[i] *= decay
		}
		for vd := range hv {
			var kv float64
			for kd := range hk {
				kv += S[kd*hv+vd] * k[kd]
			}
			write := beta * (v[vd] - kv)
			for kd := range hk {
				S[kd*hv+vd] += k[kd] * write
			}
			for kd := range hk {
				out[h*hv+vd] += S[kd*hv+vd] * q[kd]
			}
		}
	}
	return out
}

// DeltaNetRecurrence matches the plain recurrence over several tokens, output and state, on head shapes that reach every path:
// dimensions that are multiples of 4 (the unrolled loops), odd ones with grouped value heads (each loop's scalar tail), the
// NegEigval write gate, and heads larger than its stack buffers (keyDim past 512, a value head past 256), where it allocates.
func TestDeltaNetRecurrence_matchesThePlainRecurrence(t *testing.T) {
	for _, p := range []DeltaNetParams{
		{NumKeyHeads: 1, NumValueHeads: 1, KeyHeadDim: 4, ValueHeadDim: 8},
		{NumKeyHeads: 2, NumValueHeads: 4, KeyHeadDim: 3, ValueHeadDim: 5},
		{NumKeyHeads: 1, NumValueHeads: 2, KeyHeadDim: 5, ValueHeadDim: 7, NegEigval: true},
		{NumKeyHeads: 3, NumValueHeads: 3, KeyHeadDim: 200, ValueHeadDim: 300},
	} {
		t.Run(fmt.Sprintf("nk%d_nv%d_hk%d_hv%d_neg%v", p.NumKeyHeads, p.NumValueHeads, p.KeyHeadDim, p.ValueHeadDim, p.NegEigval), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(1, uint64(p.KeyHeadDim*1000+p.ValueHeadDim)))
			nv, keyDim := p.NumValueHeads, p.KeyHeadDim*p.NumKeyHeads
			vec := func(n int, scale float64) ([]float32, []float64) {
				f, d := make([]float32, n), make([]float64, n)
				for i := range f {
					f[i] = float32((rng.Float64()*2 - 1) * scale)
					d[i] = float64(f[i])
				}
				return f, d
			}
			dtBias, dtBias64 := vec(nv, 0.5)
			negExpA, negExpA64 := vec(nv, 1)
			for i := range negExpA { // negExpA = −exp(A_log) < 0, so the state decays
				negExpA[i] = -float32(math.Abs(float64(negExpA[i]))) - 0.1
				negExpA64[i] = float64(negExpA[i])
			}
			s := make([]float32, nv*p.KeyHeadDim*p.ValueHeadDim)
			s64 := make([]float64, len(s))
			for step := range 3 {
				conv, conv64 := vec(2*keyDim+nv*p.ValueHeadDim, 1)
				at, at64 := vec(nv, 2)
				bt, bt64 := vec(nv, 2)
				core := make([]float32, nv*p.ValueHeadDim) // the caller clears core; the recurrence accumulates into it
				DeltaNetRecurrence(core, conv, at, bt, dtBias, negExpA, p, s, nil)
				want := refRecurrence(conv64, at64, bt64, dtBias64, negExpA64, p, s64)
				for i := range core {
					if d := math.Abs(float64(core[i]) - want[i]); d > 1e-4*(1+math.Abs(want[i])) {
						t.Fatalf("step %d: out[%d] = %g, the plain recurrence %g", step, i, core[i], want[i])
					}
				}
				for i := range s {
					if d := math.Abs(float64(s[i]) - s64[i]); d > 1e-4*(1+math.Abs(s64[i])) {
						t.Fatalf("step %d: state[%d] = %g, the plain recurrence %g", step, i, s[i], s64[i])
					}
				}
			}
		})
	}
}

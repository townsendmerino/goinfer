// Package mamba2 holds the Mamba-2 selective state-space mixer between its two projections (the depthwise causal conv,
// the selective-scan recurrence, and the grouped gated RMSNorm) shared by decoder and the backend kernel tests (SB1,
// docs/tasks/task-subsystems-2026-10.md §8). It mirrors the HF GraniteMoeHybridMambaLayer single-token (cache) path.
// Keeping the reference here lets metal/kernels test against the single production implementation without importing
// decoder.
//
// Invariants, each with the test that holds it:
//
//	the mixer matches the HF Mamba-2 layer                decoder's TestMamba2_parity
//	Metal conv, scan and norm chain matches the CPU       metal's TestMambaKernels_cpuParity
//	Metal grouped gated norm matches the CPU              metal/kernels's TestMambaGNorm_groups
package mamba2

import (
	"math"

	"github.com/townsendmerino/goinfer/internal/hybridattn"
)

// Params is the mixer's geometry: nHeads heads of headDim (P), an SSM state of dState (N) per head-channel, B and C
// shared across the heads of each of nGroups groups, and a causal conv of DConv (K) taps. NormGroups is the gated
// norm's own group count, which families set differently: Granite normalizes over the full inner width (1),
// Nemotron-H per group (NGroups). 0 is treated as 1.
type Params struct {
	NHeads     int
	HeadDim    int // P
	DState     int // N
	NGroups    int
	DConv      int // K
	NormGroups int
}

// DInner is the inner width, NHeads·HeadDim.
func (p Params) DInner() int { return p.NHeads * p.HeadDim }

// ConvDim is the conv's width: x (DInner) and B and C (each NGroups·DState).
func (p Params) ConvDim() int { return p.DInner() + 2*p.NGroups*p.DState }

// Weights are the mixer's f32 tensors between the projections.
type Weights struct {
	ConvW  []float32 // [convDim, K] (HF conv1d.weight [convDim,1,K] flattened)
	ConvB  []float32 // [convDim]
	ALog   []float32 // [nHeads] (A = -exp(ALog))
	D      []float32 // [nHeads] skip
	DtBias []float32 // [nHeads]
	NormW  []float32 // [dInner] gated RMSNorm weight
}

// State is one layer's recurrent state: the conv window (the last K-1 xBC vectors, oldest first) and the SSM state
// [nHeads·P·N].
type State struct {
	ConvWin [][]float32
	SSM     []float32
}

// NewState returns a zero state for p.
func NewState(p Params) *State {
	return &State{SSM: make([]float32, p.NHeads*p.HeadDim*p.DState)}
}

// Mix advances one token through the mixer, reading and updating st, and returns the gated-norm output [dInner], the
// out_proj's input. proj is the in_proj output [z (dInner) ‖ xBC (convDim) ‖ dt (nHeads)]; st's conv window keeps a
// reference to its xBC part, so the caller must not reuse proj's backing array. f32 computes exp(dt·A) and the norm's
// accumulation in f32, as the GPU kernels do (a precision-localization seam, off in production).
func Mix(proj []float32, w Weights, p Params, eps float64, st *State, f32 bool) []float32 {
	dInner, convDim, K := p.DInner(), p.ConvDim(), p.DConv
	N, P := p.DState, p.HeadDim
	gSize := p.NGroups * N
	repeat := p.NHeads / p.NGroups

	z := proj[:dInner]
	xBC := proj[dInner : dInner+convDim]
	dt := proj[dInner+convDim:] // [nHeads]

	// Depthwise causal conv over xBC (+ bias, + SiLU). Taps t-K+1..t: the last K-1 come from the window (zero-padded
	// early), the K-th is this token.
	conv := make([]float32, convDim)
	win := st.ConvWin
	for c := range convDim {
		s := w.ConvB[c] + w.ConvW[c*K+(K-1)]*xBC[c] // current token (tap K-1)
		for j := 0; j < K-1; j++ {
			if idx := len(win) - (K - 1) + j; idx >= 0 {
				s += w.ConvW[c*K+j] * win[idx][c]
			}
		}
		conv[c] = hybridattn.Silu(s)
	}
	st.ConvWin = append(st.ConvWin, xBC)
	if len(st.ConvWin) > K-1 {
		st.ConvWin = st.ConvWin[len(st.ConvWin)-(K-1):]
	}

	// Split conv → x (dInner), B, C (each nGroups·N).
	x := conv[:dInner]
	B := conv[dInner : dInner+gSize]
	C := conv[dInner+gSize : dInner+2*gSize]

	// Selective SSM recurrence per head; the state persists in st.SSM.
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
			dA = float32(math.Exp(float64(dth) * A)) // f64 product (CPU reference)
		}
		S := st.SSM[head*P*N : (head+1)*P*N] // [P, N]
		out := y[head*P : (head+1)*P]
		for pi := range P {
			var acc float32
			row := S[pi*N : pi*N+N]
			dx := dth * xh[pi]
			_ = row[N-1]
			_ = Bh[N-1]
			_ = Ch[N-1]
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

	// Gated RMSNorm: y·SiLU(z), normalized per norm group, times the weight.
	gated := make([]float32, dInner)
	for i := range dInner {
		gated[i] = y[i] * hybridattn.Silu(z[i])
	}
	GatedRMSNormGrouped(gated, w.NormW, p.NormGroups, eps, f32)
	return gated
}

// GatedRMSNormGrouped applies the Mamba-2 gated RMSNorm in place: x already holds y·SiLU(gate); it is split into nGroups
// contiguous groups (group_size = len/nGroups), each RMS-normalized independently, then scaled by weight[i]. nGroups 1
// (or less) is one norm over the whole vector. f32 accumulates in f32, as the GPU gated-norm kernels do.
func GatedRMSNormGrouped(x, weight []float32, nGroups int, eps float64, f32 bool) {
	if nGroups < 1 {
		nGroups = 1
	}
	groupSize := len(x) / nGroups
	for g := 0; g < nGroups; g++ {
		seg := x[g*groupSize : (g+1)*groupSize]
		var inv float32
		if f32 {
			var ss float32
			for _, v := range seg {
				ss += v * v
			}
			inv = float32(1 / math.Sqrt(float64(ss/float32(groupSize))+eps))
		} else {
			var ss float64
			for _, v := range seg {
				ss += float64(v) * float64(v)
			}
			inv = float32(1 / math.Sqrt(ss/float64(groupSize)+eps))
		}
		for j := range seg {
			seg[j] = weight[g*groupSize+j] * seg[j] * inv
		}
	}
}

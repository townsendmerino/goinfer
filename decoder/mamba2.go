package decoder

import (
	"math"
	"os"

	"github.com/townsendmerino/aikit/linalg"
)

// Mamba-2 selective state-space sequence mixer (Granite-4.0 hybrid). Like Gated
// DeltaNet (deltanet.go) it is an inherently sequential recurrence driven one token
// at a time by both prefill and decode, with a small per-token causal conv before
// the scan. This is the parity-first SEQUENTIAL form — the algebraic oracle a
// chunked/segsum scan (a later prefill optimization) is proven equivalent to. It
// mirrors the HF GraniteMoeHybridMambaLayer single-token (cache) path exactly.
//
// Shapes: nHeads heads of headDim (P), an SSM state of dState (N) per head-channel,
// B/C shared across heads within a group (nGroups). The in_proj emits
// [gate z (dInner) ‖ xBC (convDim) ‖ dt (nHeads)]; conv1d runs over xBC then splits
// into x (dInner) ‖ B (nGroups·N) ‖ C (nGroups·N). dInner = nHeads·P.

type mamba2Params struct {
	NHeads  int
	HeadDim int // P
	DState  int // N
	NGroups int
	DConv   int // K (causal conv kernel)
	Hidden  int
	// NormGroups is the group count for the GATED RMSNorm specifically — families
	// differ here: Granite (GraniteMoeHybridRMSNormGated) normalizes over the FULL
	// dInner (NormGroups 1), Nemotron-H (Zamba2RMSNormGated) normalizes per group
	// (NormGroups = NGroups). Equal when NGroups==1. 0 ⇒ treated as 1.
	NormGroups int
}

func (p mamba2Params) dInner() int  { return p.NHeads * p.HeadDim }
func (p mamba2Params) convDim() int { return p.dInner() + 2*p.NGroups*p.DState }
func (p mamba2Params) projDim() int { return 2*p.dInner() + 2*p.NGroups*p.DState + p.NHeads }

// mamba2Weights holds one mixer's tensors, all f32 (parity-first, like the qwen35
// DeltaNet/softmax weights). Row-major [out, in] for the projections.
type mamba2Weights struct {
	inProj  []float32 // [projDim, hidden]
	convW   []float32 // [convDim, K] (HF conv1d.weight [convDim,1,K] flattened)
	convB   []float32 // [convDim]
	aLog    []float32 // [nHeads] (A = -exp(aLog))
	d       []float32 // [nHeads] skip
	dtBias  []float32 // [nHeads]
	normW   []float32 // [dInner] gated RMSNorm weight
	outProj []float32 // [hidden, dInner]
}

// mamba2State is the per-layer recurrent state: the short causal conv window (last
// K-1 xBC vectors, oldest first) plus the SSM state [nHeads · P · N]. A second kind
// of recurrent state slotting into the hybrid cache alongside deltaState.
type mamba2State struct {
	convWin [][]float32
	ssm     []float32
}

func newMamba2State(p mamba2Params) *mamba2State {
	return &mamba2State{ssm: make([]float32, p.NHeads*p.HeadDim*p.DState)}
}

// mamba2Step advances one token through the Mamba-2 mixer, reading and updating st,
// returning the layer output [hidden].
func mamba2Step(h []float32, w *mamba2Weights, p mamba2Params, eps float64, st *mamba2State) []float32 {
	dInner, convDim, K := p.dInner(), p.convDim(), p.DConv
	N, P := p.DState, p.HeadDim
	gSize := p.NGroups * N
	repeat := p.NHeads / p.NGroups

	// 1. in_proj → [z (gate), xBC, dt].
	inProj, outProj := w.inProj, w.outProj
	if ssmQ8CPU { // confirmation seam: match the resident W8A8 — int8 weights AND int8 activations
		inProj = q8RoundTrip(inProj, p.projDim(), p.Hidden)
		outProj = q8RoundTrip(outProj, p.Hidden, dInner)
		h = q8RoundTrip(h, 1, len(h)) // the in_proj activation (resident rmsQuant → int8)
	}
	proj := matvec(inProj, p.projDim(), p.Hidden, h)
	z := proj[:dInner]
	xBC := proj[dInner : dInner+convDim]
	dt := proj[dInner+convDim:] // [nHeads]

	// 2. Depthwise causal conv over xBC (+ bias, + SiLU). Taps t-K+1..t: the last
	// K-1 come from convWin (zero-padded early), the K-th is this token.
	conv := make([]float32, convDim)
	win := st.convWin
	for c := range convDim {
		s := w.convB[c] + w.convW[c*K+(K-1)]*xBC[c] // current token (tap K-1)
		for j := 0; j < K-1; j++ {
			if idx := len(win) - (K - 1) + j; idx >= 0 {
				s += w.convW[c*K+j] * win[idx][c]
			}
		}
		conv[c] = silu(s)
	}
	st.convWin = append(st.convWin, xBC)
	if len(st.convWin) > K-1 {
		st.convWin = st.convWin[len(st.convWin)-(K-1):]
	}

	// 3. Split conv → x (dInner), B, C (each nGroups·N).
	x := conv[:dInner]
	B := conv[dInner : dInner+gSize]
	C := conv[dInner+gSize : dInner+2*gSize]

	// 4. Selective SSM recurrence per head; state persists in st.ssm.
	y := make([]float32, dInner)
	for head := range p.NHeads {
		g := head / repeat
		Bh := B[g*N : (g+1)*N]
		Ch := C[g*N : (g+1)*N]
		xh := x[head*P : (head+1)*P]
		A := -math.Exp(float64(w.aLog[head]))
		dth := softplusf(dt[head] + w.dtBias[head])
		dA := float32(math.Exp(float64(dth) * A))
		S := st.ssm[head*P*N : (head+1)*P*N] // [P, N]
		out := y[head*P : (head+1)*P]
		for pi := range P {
			var acc float32
			row := S[pi*N : pi*N+N]
			dx := dth * xh[pi]
			for n := range N {
				row[n] = row[n]*dA + dx*Bh[n]
				acc += row[n] * Ch[n]
			}
			out[pi] = acc + w.d[head]*xh[pi]
		}
	}

	// 5. Gated RMSNorm (y · SiLU(z), normalized × weight), then out_proj. The norm is
	// GROUPED by NGroups (group_size = dInner/NGroups): each group normalizes
	// independently (Zamba2/Mamba-2 RMSNormGated). NGroups=1 ⇒ a single flat norm.
	gated := make([]float32, dInner)
	for i := range dInner {
		gated[i] = y[i] * silu(z[i])
	}
	gatedRMSNormGrouped(gated, w.normW, p.NormGroups, eps)
	if ssmQ8CPU {
		gated = q8RoundTrip(gated, 1, len(gated)) // the out_proj activation (resident quant → int8)
	}
	return matvec(outProj, p.Hidden, dInner, gated)
}

// ssmQ8CPU + q8RoundTrip are a confirmation seam (GOINFER_SSM_Q8CPU): round-trip the
// Mamba projections through int8 so the CPU reference carries the SAME quantization error
// as the resident W8A8 path — isolating kernel-correctness from int8 sensitivity.
var ssmQ8CPU = os.Getenv("GOINFER_SSM_Q8CPU") != ""

func q8RoundTrip(w []float32, N, K int) []float32 {
	q, s := linalg.QuantizeRowsInt8(w, N, K)
	out := make([]float32, N*K)
	for r := 0; r < N; r++ {
		for k := 0; k < K; k++ {
			out[r*K+k] = float32(q[r*K+k]) * s[r]
		}
	}
	return out
}

// gatedRMSNormGrouped applies the Mamba-2 gated RMSNorm in place: x already holds
// y·SiLU(gate); it is split into nGroups contiguous groups (group_size = len/nGroups)
// and each group is RMS-normalized independently, then scaled by weight[i]. nGroups=1
// is a single flat RMSNorm over the whole vector.
func gatedRMSNormGrouped(x, weight []float32, nGroups int, eps float64) {
	if nGroups < 1 {
		nGroups = 1
	}
	groupSize := len(x) / nGroups
	for g := 0; g < nGroups; g++ {
		seg := x[g*groupSize : (g+1)*groupSize]
		var ss float64
		for _, v := range seg {
			ss += float64(v) * float64(v)
		}
		inv := float32(1 / math.Sqrt(ss/float64(groupSize)+eps))
		for j := range seg {
			seg[j] = weight[g*groupSize+j] * seg[j] * inv
		}
	}
}

// mamba2Seq runs the mixer over a whole sequence from a fresh state — a thin loop
// over mamba2Step, the form the parity test exercises (the forward path uses the
// same step with a cached state).
func mamba2Seq(h [][]float32, w *mamba2Weights, p mamba2Params, eps float64) [][]float32 {
	st := newMamba2State(p)
	out := make([][]float32, len(h))
	for t := range h {
		out[t] = mamba2Step(h[t], w, p, eps, st)
	}
	return out
}

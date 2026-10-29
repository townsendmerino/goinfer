// Package hybridattn holds the hybrid-attention layers' operations (the Gated-DeltaNet linear-attention recurrence,
// its L2 head normalization, and the Qwen 3.5 interleaved query/gate split and sigmoid gate) shared by decoder and the
// backend kernel tests (SB1, docs/tasks/task-subsystems-2026-10.md §8). Keeping the reference here lets metal/kernels
// test against the single production implementation without importing decoder.
//
// Invariants, each with the test that holds it:
//
//	DeltaNet recurrence matches the frozen reference      decoder's TestGatedDeltaNet_parity
//	batched prefill matches the single-token step         decoder's TestDeltaNetCoreN_bitIdentical
//	Metal DeltaNet kernels match the CPU reference        metal/kernels's TestDeltaNetKernels_cpuParity
//	Metal q-gate kernels match the CPU reference          metal/kernels's TestQGateKernels_cpuParity
package hybridattn

import (
	"math"
	"sync/atomic"
	"time"

	"github.com/townsendmerino/aikit/linalg"
)

// DeltaNetParams holds the head geometry and variant flags for one Gated DeltaNet layer.
type DeltaNetParams struct {
	ConvKernel    int     // depthwise causal conv width over [q;k;v] (linear_conv_kernel_dim)
	KeyHeadDim    int     // per-head key/query dim (linear_key_head_dim)
	ValueHeadDim  int     // per-head value dim (linear_value_head_dim)
	NumKeyHeads   int     // linear_num_key_heads
	NumValueHeads int     // linear_num_value_heads (GVA: a multiple of NumKeyHeads)
	NegEigval     bool    // Olmo Hybrid: widen sigmoid's [0,1) write-gate to [0,2)
	ONormEps      float64 // Olmo Hybrid: override the output gated-RMSNorm epsilon when non-zero
}

// DeltaNetWeights holds one Gated DeltaNet layer's parameters, row-major, as HF stores them.
type DeltaNetWeights struct {
	InProjQKV linalg.WeightMat // [convDim, hidden]   → [q;k;v]
	InProjZ   linalg.WeightMat // [valueDim, hidden]  → output gate z
	InProjB   []float32        // [numV, hidden]      → write-gate logits β
	InProjA   []float32        // [numV, hidden]      → decay-gate input a
	ConvW     []float32        // [convDim, K]        depthwise causal conv ([convDim,1,K] flattened)
	DtBias    []float32        // [numV]
	NegExpA   []float32        // [numV]              −exp(A_log), precomputed (see NegExpAFromLog)
	NormW     []float32        // [headVDim]          gated RMSNorm weight
	OutProj   linalg.WeightMat // [hidden, valueDim]
}

// DeltaState is one Gated DeltaNet layer's streaming state: the last K-1 conv inputs and the recurrent matrix state S
// (one [head_k_dim, head_v_dim] block per value head).
type DeltaState struct {
	ConvWin [][]float32 // up to K-1 prior mixed_qkv vectors, oldest first
	S       []float32   // [numV * head_k_dim * head_v_dim]
}

// NewDeltaState allocates a zeroed streaming state for p's geometry.
func NewDeltaState(p DeltaNetParams) *DeltaState {
	return &DeltaState{S: make([]float32, p.NumValueHeads*p.KeyHeadDim*p.ValueHeadDim)}
}

// DeltaNet pairs a Gated DeltaNet layer's weights and geometry for standalone CPU reference steps.
type DeltaNet struct {
	Weights DeltaNetWeights
	Params  DeltaNetParams
}

// NewDeltaNet wraps raw float32 weight slices into a DeltaNet and fresh DeltaState so a backend kernel test can drive
// GatedDeltaNetStep without importing decoder.
func NewDeltaNet(
	convK, hk, hv, nk, nv, hidden int,
	inProjQKV, inProjZ, inProjB, inProjA, convW, dtBias, negExpA, normW, outProj []float32,
) (*DeltaNet, *DeltaState) {
	p := DeltaNetParams{
		ConvKernel:    convK,
		KeyHeadDim:    hk,
		ValueHeadDim:  hv,
		NumKeyHeads:   nk,
		NumValueHeads: nv,
	}
	convDim := 2*hk*nk + hv*nv
	valueDim := hv * nv
	w := DeltaNetWeights{
		InProjQKV: linalg.WrapF32(inProjQKV, convDim, hidden),
		InProjZ:   linalg.WrapF32(inProjZ, valueDim, hidden),
		InProjB:   inProjB,
		InProjA:   inProjA,
		ConvW:     convW,
		DtBias:    dtBias,
		NegExpA:   negExpA,
		NormW:     normW,
		OutProj:   linalg.WrapF32(outProj, hidden, valueDim),
	}
	return &DeltaNet{Weights: w, Params: p}, NewDeltaState(p)
}

// DeltaNetStep advances one token through dn on the CPU using f32 projections.
func DeltaNetStep(h []float32, dn *DeltaNet, hidden int, eps float64, st *DeltaState) []float32 {
	w := &dn.Weights
	return GatedDeltaNetStep(
		nil, h,
		&w.InProjQKV, &w.InProjZ, &w.OutProj,
		w.InProjB, w.InProjA, w.ConvW, w.DtBias, w.NegExpA, w.NormW,
		dn.Params, hidden, eps,
		st.S, &st.ConvWin, nil,
	)
}

// DeltaNetTimers holds optional atomic accumulators for decoder's GOINFER_DELTANET_TIMING diagnostic.
type DeltaNetTimers struct {
	ProjNs     *atomic.Int64
	RecurNs    *atomic.Int64
	OtherNs    *atomic.Int64
	StateElems *atomic.Int64
	Calls      *atomic.Int64
}

// NegExpAFromLog precomputes the DeltaNet decay coefficient −exp(A_log) that the recurrence multiplies by
// softplus(a+dt_bias).
func NegExpAFromLog(aLog []float32) []float32 {
	out := make([]float32, len(aLog))
	for i, v := range aLog {
		out[i] = -float32(math.Exp(float64(v)))
	}
	return out
}

// Matvec computes y[r] = Σ_c W[r*cols+c]·x[c] for a row-major [rows,cols] weight via the SIMD A·Bᵀ kernel at M=1.
func Matvec(w []float32, rows, cols int, x []float32) []float32 {
	y := make([]float32, rows)
	linalg.MatmulBT(x, w, y, 1, cols, rows)
	return y
}

// Sigmoidf returns 1 / (1 + exp(-x)) computed in float64 and rounded back to float32.
func Sigmoidf(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

// Softplusf matches torch.nn.functional.softplus (default beta=1, threshold=20: linear above the threshold for
// numerical stability).
func Softplusf(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

// Silu returns x * sigmoid(x) computed in float64 and rounded back to float32.
func Silu(x float32) float32 {
	x64 := float64(x)
	return float32(x64 / (1 + math.Exp(-x64)))
}

// AddScaled computes dst[i] += scale * src[i] across len(dst).
func AddScaled(dst, src []float32, scale float32) {
	if len(dst) == 0 {
		return
	}
	_ = dst[len(dst)-1]
	_ = src[len(dst)-1]
	i := 0
	for ; i+3 < len(dst); i += 4 {
		dst[i] += scale * src[i]
		dst[i+1] += scale * src[i+1]
		dst[i+2] += scale * src[i+2]
		dst[i+3] += scale * src[i+3]
	}
	for ; i < len(dst); i++ {
		dst[i] += scale * src[i]
	}
}

// GatedDeltaNetStep advances one token through the Gated DeltaNet layer, reading and updating s and convWin, and
// returns the layer output [hidden]. When proj is nil, the three WeightMat projections run as f32 Matvec and panic if
// any WeightMat is not f32.
func GatedDeltaNetStep(
	proj func(w *linalg.WeightMat, x []float32, pre bool) []float32,
	h []float32,
	inProjQKV, inProjZ, outProj *linalg.WeightMat,
	inProjB, inProjA, convW, dtBias, negExpA, normW []float32,
	p DeltaNetParams,
	hidden int,
	eps float64,
	s []float32,
	convWin *[][]float32,
	timers *DeltaNetTimers,
) []float32 {
	if proj == nil {
		proj = func(wm *linalg.WeightMat, x []float32, _ bool) []float32 {
			f32, ok := wm.F32()
			if !ok {
				panic("program.GatedDeltaNetStep: nil proj requires an f32 WeightMat (got " + wm.Kind() + ")")
			}
			return Matvec(f32, wm.Rows(), wm.Cols(), x)
		}
	}
	nv := p.NumValueHeads
	var t0 time.Time
	if timers != nil {
		t0 = time.Now()
	}
	mixed := proj(inProjQKV, h, true)
	if timers != nil {
		timers.ProjNs.Add(int64(time.Since(t0)))
		t0 = time.Now()
	}
	bt := Matvec(inProjB, nv, hidden, h)
	at := Matvec(inProjA, nv, hidden, h)
	z := proj(inProjZ, h, true)
	if timers != nil {
		timers.OtherNs.Add(int64(time.Since(t0)))
	}
	core := make([]float32, p.ValueHeadDim*nv)
	DeltaNetCore(core, mixed, bt, at, z, convW, dtBias, negExpA, normW, p, eps, s, convWin, timers)
	if timers != nil {
		t0 = time.Now()
	}
	out := proj(outProj, core, false)
	if timers != nil {
		timers.ProjNs.Add(int64(time.Since(t0)))
		timers.Calls.Add(1)
	}
	return out
}

// DeltaNetCore is one token's sequential middle of the Gated DeltaNet layer, from the projections' outputs to
// out_proj's input: the depthwise causal conv (+ SiLU) over convWin, the gated delta-rule recurrence (state in s),
// and the gated RMSNorm (× SiLU(z)). mixed is retained in convWin, so the caller must not reuse its storage.
func DeltaNetCore(
	core, mixed, bt, at, z []float32,
	convW, dtBias, negExpA, normW []float32,
	p DeltaNetParams,
	eps float64,
	s []float32,
	convWin *[][]float32,
	timers *DeltaNetTimers,
) {
	nk, nv := p.NumKeyHeads, p.NumValueHeads
	hk, hv := p.KeyHeadDim, p.ValueHeadDim
	keyDim, valueDim := hk*nk, hv*nv
	convDim := 2*keyDim + valueDim
	K := p.ConvKernel
	onEps := eps
	if p.ONormEps != 0 {
		onEps = p.ONormEps
	}

	var t0 time.Time
	if timers != nil {
		t0 = time.Now()
	}

	// 1. Depthwise causal conv (+ SiLU). Taps t-K+1..t: the last K-1 come from
	// convWin (zero-padded early), the K-th is this token.
	conv := make([]float32, convDim)
	win := *convWin
	for c := range convDim {
		sc := convW[c*K+(K-1)] * mixed[c] // j=K-1 tap = current token
		for j := 0; j < K-1; j++ {
			if idx := len(win) - (K - 1) + j; idx >= 0 {
				sc += convW[c*K+j] * win[idx][c]
			}
		}
		conv[c] = Silu(sc)
	}
	// slide the window: keep the last K-1 mixed vectors
	*convWin = append(*convWin, mixed)
	if len(*convWin) > K-1 {
		*convWin = (*convWin)[len(*convWin)-(K-1):]
	}
	if timers != nil {
		timers.OtherNs.Add(int64(time.Since(t0))) // conv; the gate projections are timed by the caller
	}

	// 2. Gated delta-rule recurrence, per value head; state persists in s.
	clear(core)
	DeltaNetRecurrence(core, conv, at, bt, dtBias, negExpA, p, s, timers)

	var capPre, capGate []float32
	if deltaCapHook != nil { // test seam: step 2's output, before step 3 overwrites it in place
		capPre = append([]float32(nil), core...)
		capGate = make([]float32, 2*nv)
		for headV := range nv {
			g := negExpA[headV] * Softplusf(at[headV]+dtBias[headV])
			capBeta := Sigmoidf(bt[headV])
			if p.NegEigval {
				capBeta *= 2
			}
			capGate[headV*2+0] = capBeta
			capGate[headV*2+1] = float32(math.Exp(float64(g)))
		}
	}

	// 3. Gated RMSNorm (over head_v_dim, × SiLU(z)); out_proj is the caller's.
	if timers != nil {
		t0 = time.Now()
	}
	for headV := range nv {
		seg := core[headV*hv : headV*hv+hv]
		zt := z[headV*hv : headV*hv+hv]
		var ss float64
		for _, x := range seg {
			ss += float64(x) * float64(x)
		}
		inv := float32(1 / math.Sqrt(ss/float64(hv)+onEps))
		for vd := range hv {
			seg[vd] = seg[vd] * inv * normW[vd] * Silu(zt[vd])
		}
	}
	if timers != nil {
		timers.OtherNs.Add(int64(time.Since(t0)))
	}
	if deltaCapHook != nil {
		deltaCapHook(mixed, conv, append(append([]float32(nil), bt...), at...), capGate, capPre, core, z)
	}
}

// DeltaNetRecurrence runs one token's per-head gated delta-rule recurrence over s.
func DeltaNetRecurrence(
	core, conv, at, bt, dtBias, negExpA []float32,
	p DeltaNetParams,
	s []float32,
	timers *DeltaNetTimers,
) {
	nk, nv := p.NumKeyHeads, p.NumValueHeads
	hk, hv := p.KeyHeadDim, p.ValueHeadDim
	keyDim := hk * nk
	rep := nv / nk
	qScale := float32(1 / math.Sqrt(float64(hk)))

	var t0 time.Time
	if timers != nil {
		t0 = time.Now()
	}

	// 1. Precompute q and k L2-normalized projections across the nk key heads.
	// In Gated DeltaNet, q and k depend only on headK = headV / rep.
	// Precomputing avoids recomputing them rep times and avoids 2*nv heap allocations.
	var qAllBuf, kAllBuf [512]float32
	var qAll, kAll []float32
	if keyDim <= len(qAllBuf) {
		qAll = qAllBuf[:keyDim]
		kAll = kAllBuf[:keyDim]
	} else {
		qAll = make([]float32, keyDim)
		kAll = make([]float32, keyDim)
	}
	for headK := range nk {
		L2NormScaledInto(qAll[headK*hk:(headK+1)*hk], conv[headK*hk:(headK+1)*hk], qScale)
		L2NormScaledInto(kAll[headK*hk:(headK+1)*hk], conv[keyDim+headK*hk:keyDim+(headK+1)*hk], 1)
	}

	// 2. Stack buffer for kv / delta to eliminate nv allocations per step.
	var kvBuf [256]float32
	var kv []float32
	if hv <= len(kvBuf) {
		kv = kvBuf[:hv]
	} else {
		kv = make([]float32, hv)
	}

	for headV := range nv {
		if timers != nil {
			t0 = time.Now()
		}
		headK := headV / rep
		q := qAll[headK*hk : (headK+1)*hk]
		k := kAll[headK*hk : (headK+1)*hk]
		v := conv[2*keyDim+headV*hv : 2*keyDim+headV*hv+hv]
		g := negExpA[headV] * Softplusf(at[headV]+dtBias[headV]) // log-decay (negExpA = −exp(A_log))
		gt := float32(math.Exp(float64(g)))
		beta := Sigmoidf(bt[headV])
		if p.NegEigval {
			beta *= 2 // Olmo Hybrid: widen sigmoid's [0,1) write-gate to [0,2)
		}
		if timers != nil {
			timers.OtherNs.Add(int64(time.Since(t0))) // gate scalars
			t0 = time.Now()
		}

		S := s[headV*hk*hv : (headV+1)*hk*hv] // [hk, hv]
		sLen := len(S)
		_ = S[sLen-1]
		si := 0
		for ; si+3 < sLen; si += 4 {
			S[si] *= gt
			S[si+1] *= gt
			S[si+2] *= gt
			S[si+3] *= gt
		}
		for ; si < sLen; si++ {
			S[si] *= gt
		}

		out := core[headV*hv : headV*hv+hv]
		clear(kv)
		for kd := range hk {
			row := S[kd*hv : kd*hv+hv]
			kk := k[kd]
			AddScaled(kv, row, kk)
		}

		delta := kv // aliased and overwritten in place: index vd is read once, then written once
		_ = delta[hv-1]
		_ = v[hv-1]
		vdi := 0
		for ; vdi+3 < hv; vdi += 4 {
			delta[vdi] = (v[vdi] - delta[vdi]) * beta
			delta[vdi+1] = (v[vdi+1] - delta[vdi+1]) * beta
			delta[vdi+2] = (v[vdi+2] - delta[vdi+2]) * beta
			delta[vdi+3] = (v[vdi+3] - delta[vdi+3]) * beta
		}
		for ; vdi < hv; vdi++ {
			delta[vdi] = (v[vdi] - delta[vdi]) * beta
		}

		_ = out[hv-1]
		for kd := range hk {
			row := S[kd*hv : kd*hv+hv]
			kk, qq := k[kd], q[kd]
			_ = row[hv-1]
			_ = delta[hv-1]
			vd := 0
			for ; vd+3 < hv; vd += 4 {
				r0 := row[vd] + kk*delta[vd]
				r1 := row[vd+1] + kk*delta[vd+1]
				r2 := row[vd+2] + kk*delta[vd+2]
				r3 := row[vd+3] + kk*delta[vd+3]
				row[vd] = r0
				row[vd+1] = r1
				row[vd+2] = r2
				row[vd+3] = r3
				out[vd] += r0 * qq
				out[vd+1] += r1 * qq
				out[vd+2] += r2 * qq
				out[vd+3] += r3 * qq
			}
			for ; vd < hv; vd++ {
				r := row[vd] + kk*delta[vd]
				row[vd] = r
				out[vd] += r * qq
			}
		}
		if timers != nil {
			timers.RecurNs.Add(int64(time.Since(t0)))
			timers.StateElems.Add(int64(hk * hv))
		}
	}
}

// deltaCapHook hands a backend kernel test every intermediate its kernels must reproduce per step.
var deltaCapHook func(mixed, conv, gateIn, betaGate, corePre, gated, z []float32)

// SetDeltaCapHook installs (or clears, with nil) the DeltaNet intermediate-capture hook.
func SetDeltaCapHook(f func(mixed, conv, gateIn, betaGate, corePre, gated, z []float32)) {
	deltaCapHook = f
}

// HasDeltaCapHook reports whether a DeltaNet intermediate-capture hook is currently installed.
func HasDeltaCapHook() bool {
	return deltaCapHook != nil
}

// L2NormScaledInto writes x/‖x‖ (eps 1e-6, matching FLA's l2norm) times s into dst.
func L2NormScaledInto(dst, x []float32, s float32) {
	n := len(x)
	if n == 0 {
		return
	}
	_ = dst[n-1]
	_ = x[n-1]
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(math.Sqrt(1/(ss+1e-6))) * s
	i := 0
	for ; i+3 < n; i += 4 {
		dst[i] = x[i] * inv
		dst[i+1] = x[i+1] * inv
		dst[i+2] = x[i+2] * inv
		dst[i+3] = x[i+3] * inv
	}
	for ; i < n; i++ {
		dst[i] = x[i] * inv
	}
}

// L2NormScaled returns x/‖x‖ (eps 1e-6, matching FLA's l2norm) times s.
func L2NormScaled(x []float32, s float32) []float32 {
	out := make([]float32, len(x))
	L2NormScaledInto(out, x, s)
	return out
}

// SplitQGate splits one position's double-width q_proj output into q and gate. The layout is [query ‖ gate] PER HEAD,
// interleaved, not two concatenated blocks. Metal's delta_qsplit mirrors it (metal/kernels/qgate_kernels_test.go).
func SplitQGate(qg, q, gate []float32, nH, hd int) {
	for hh := range nH {
		copy(q[hh*hd:hh*hd+hd], qg[hh*2*hd:hh*2*hd+hd])
		copy(gate[hh*hd:hh*hd+hd], qg[hh*2*hd+hd:hh*2*hd+2*hd])
	}
}

// SplitQGateAlloc allocates q and gate [nH*hd] and splits qg into them via SplitQGate.
func SplitQGateAlloc(qg []float32, nH, hd int) (q, gate []float32) {
	q = make([]float32, nH*hd)
	gate = make([]float32, nH*hd)
	SplitQGate(qg, q, gate, nH, hd)
	return q, gate
}

// QGateContext is the output gate: ctx *= sigmoid(gate), elementwise and in place, before o_proj. Metal's
// delta_attn_gate mirrors it.
func QGateContext(ctx, gate []float32) {
	for i := range ctx {
		ctx[i] *= Sigmoidf(gate[i])
	}
}

package decoder

import "math"

// Chunked Gated DeltaNet scan — a parallel-friendly reformulation of the
// sequential recurrence in gatedDeltaNet, processing the position dimension in
// chunks. It is algebraically equivalent to the per-step recurrence (proven by
// TestGatedDeltaNet_chunkedMatchesSequential), so it is the reference kernel for
// the eventual perf rewrite (the heavy per-pair / per-chunk terms become
// matmuls). It is NOT yet wired into the forward — the qwen3_5_moe path is still
// single-token streaming — so it carries no regression risk to the proven
// sequential path.
//
// Derivation (per value head, chunk of L positions, incoming state S_in):
// unrolling S_i = gt_i·S_{i-1} + k_i⊗u_i with u_i = β_i·(v_i − (gt_i·S_{i-1})ᵀk_i)
// and cumulative decay c_i = Π_{j≤i} gt_j gives a unit-lower-triangular solve
//
//	u_i = rhs_i − β_i·Σ_{m<i} (c_i/c_m)(k_m·k_i)·u_m,   rhs_i = β_i·(v_i − c_i·(S_inᵀk_i))
//
// then out_i = c_i·(S_inᵀq_i) + Σ_{m≤i} (c_i/c_m)(k_m·q_i)·u_m and the chunk-end
// state S_out = c_{L−1}·S_in + Σ_m (c_{L−1}/c_m)·k_m⊗u_m. Because gt_i = exp(g)∈(0,1),
// c is non-increasing and every c_i/c_m ratio is ≤ 1 — numerically stable.

// gatedDeltaNetChunked runs the layer over a whole sequence from a fresh state
// using the chunked scan, returning the same [hidden] output per position as
// gatedDeltaNet. chunk is the scan block size (chunk ≤ 0 ⇒ the whole sequence;
// chunk == 1 reduces exactly to the sequential step).
func gatedDeltaNetChunked(h [][]float32, w *deltaNetWeights, p qwen35Params, hidden int, eps float64, chunk int) [][]float32 {
	n := len(h)
	if n == 0 {
		return nil
	}
	nk, nv := p.NumKeyHeads, p.NumValueHeads
	hk, hv := p.KeyHeadDim, p.ValueHeadDim
	keyDim, valueDim := hk*nk, hv*nv
	convDim := 2*keyDim + valueDim
	K := p.ConvKernel
	rep := nv / nk
	qScale := float32(1 / math.Sqrt(float64(hk)))
	if chunk <= 0 || chunk > n {
		chunk = n
	}

	// --- position-local stages (identical arithmetic to gatedDeltaNetStep) ---

	// Projection (mixed_qkv) per position, then the depthwise causal conv + SiLU.
	mixed := make([][]float32, n)
	for t := range h {
		mixed[t] = matvec(w.inProjQKV, convDim, hidden, h[t])
	}
	conv := make([][]float32, n)
	for t := range n {
		cv := make([]float32, convDim)
		for c := range convDim {
			s := w.convW[c*K+(K-1)] * mixed[t][c] // current token (tap K-1)
			for j := 0; j < K-1; j++ {
				if tt := t - (K - 1) + j; tt >= 0 {
					s += w.convW[c*K+j] * mixed[tt][c]
				}
			}
			cv[c] = silu(s)
		}
		conv[t] = cv
	}

	// Gates (β, decay gt) per value head and the output gate z, per position.
	gt := make([][]float32, n)
	beta := make([][]float32, n)
	z := make([][]float32, n)
	for t := range n {
		bt := matvec(w.inProjB, nv, hidden, h[t])
		at := matvec(w.inProjA, nv, hidden, h[t])
		z[t] = matvec(w.inProjZ, valueDim, hidden, h[t])
		gtt := make([]float32, nv)
		btt := make([]float32, nv)
		for hd := range nv {
			g := w.negExpA[hd] * softplusf(at[hd]+w.dtBias[hd])
			gtt[hd] = float32(math.Exp(float64(g)))
			btt[hd] = sigmoidf(bt[hd])
		}
		gt[t], beta[t] = gtt, btt
	}

	// --- the chunked recurrence (the actual scan), per value head ---

	core := make([][]float32, n) // pre-norm output [valueDim] per position
	for t := range n {
		core[t] = make([]float32, valueDim)
	}
	for headV := range nv {
		headK := headV / rep
		S := make([]float32, hk*hv) // [hk, hv], persists across chunks
		for start := 0; start < n; start += chunk {
			end := min(start+chunk, n)
			scanChunk(core, S, conv, gt, beta, start, end, headV, headK, hk, hv, keyDim, qScale)
		}
	}

	// --- gated RMSNorm (× SiLU(z)) then out_proj, per position ---
	out := make([][]float32, n)
	for t := range n {
		for headV := range nv {
			seg := core[t][headV*hv : headV*hv+hv]
			zt := z[t][headV*hv : headV*hv+hv]
			var ss float64
			for _, x := range seg {
				ss += float64(x) * float64(x)
			}
			inv := float32(1 / math.Sqrt(ss/float64(hv)+eps))
			for vd := range hv {
				seg[vd] = seg[vd] * inv * w.normW[vd] * silu(zt[vd])
			}
		}
		out[t] = matvec(w.outProj, hidden, valueDim, core[t])
	}
	return out
}

// scanChunk runs the gated delta-rule scan over positions [start,end) for one
// value head, updating the matrix state S in place and writing each position's
// pre-norm output into core. See the derivation above.
func scanChunk(core [][]float32, S []float32, conv, gt, beta [][]float32,
	start, end, headV, headK, hk, hv, keyDim int, qScale float32) {
	L := end - start
	// Gather the chunk's per-position q (scaled+normed), k (normed), v, decay, β.
	q := make([][]float32, L)
	k := make([][]float32, L)
	v := make([][]float32, L)
	c := make([]float32, L) // cumulative decay c_i = Π_{j≤i} gt_j within the chunk
	bta := make([]float32, L)
	for i := range L {
		t := start + i
		q[i] = l2normScaled(conv[t][headK*hk:headK*hk+hk], qScale)
		k[i] = l2normScaled(conv[t][keyDim+headK*hk:keyDim+headK*hk+hk], 1)
		v[i] = conv[t][2*keyDim+headV*hv : 2*keyDim+headV*hv+hv]
		bta[i] = beta[t][headV]
		if i == 0 {
			c[i] = gt[t][headV]
		} else {
			c[i] = c[i-1] * gt[t][headV]
		}
	}

	// u_i via forward substitution; out_i and the state carry use S_in (= S here).
	u := make([][]float32, L)
	for i := range L {
		// rhs_i = β_i·(v_i − c_i·(S_inᵀk_i))
		ui := make([]float32, hv)
		for vd := range hv {
			var sk float32 // (S_inᵀ k_i)[vd]
			for kd := range hk {
				sk += S[kd*hv+vd] * k[i][kd]
			}
			ui[vd] = bta[i] * (v[i][vd] - c[i]*sk)
		}
		// − β_i·Σ_{m<i} (c_i/c_m)(k_m·k_i)·u_m
		for m := range i {
			var kk float32
			for kd := range hk {
				kk += k[m][kd] * k[i][kd]
			}
			coef := bta[i] * (c[i] / c[m]) * kk
			um := u[m]
			for vd := range hv {
				ui[vd] -= coef * um[vd]
			}
		}
		u[i] = ui
	}

	// out_i = c_i·(S_inᵀq_i) + Σ_{m≤i} (c_i/c_m)(k_m·q_i)·u_m
	for i := range L {
		out := core[start+i][headV*hv : headV*hv+hv]
		for vd := range hv {
			var sq float32 // (S_inᵀ q_i)[vd]
			for kd := range hk {
				sq += S[kd*hv+vd] * q[i][kd]
			}
			out[vd] = c[i] * sq
		}
		for m := 0; m <= i; m++ {
			var kq float32
			for kd := range hk {
				kq += k[m][kd] * q[i][kd]
			}
			coef := (c[i] / c[m]) * kq
			um := u[m]
			for vd := range hv {
				out[vd] += coef * um[vd]
			}
		}
	}

	// S_out = c_{L−1}·S_in + Σ_m (c_{L−1}/c_m)·k_m⊗u_m
	cl := c[L-1]
	for kd := range hk {
		for vd := range hv {
			S[kd*hv+vd] *= cl
		}
	}
	for m := range L {
		scale := cl / c[m]
		um := u[m]
		km := k[m]
		for kd := range hk {
			kc := scale * km[kd]
			row := kd * hv
			for vd := range hv {
				S[row+vd] += kc * um[vd]
			}
		}
	}
}

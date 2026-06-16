package decoder

import "math"

// Chunked Mamba-2 scan — a parallel-friendly reformulation of the sequential
// recurrence in mamba2Seq, processing the position dimension in chunks. Mamba-2's
// SSM is a plain LINEAR recurrence (S_t = dA_t·S_{t-1} + dt_t·x_t⊗B_t; no DeltaNet
// β-correction / triangular solve), so the unrolling is a direct cumulative-decay
// sum. Algebraically equivalent to the per-step recurrence (proven by
// TestMamba2_chunkedMatchesSequential), so it is the reference kernel for the
// eventual prefill perf rewrite (the per-pair terms become matmuls). NOT yet wired
// into the forward — runLayersGranite is still single-token streaming — so it carries
// no regression risk to the proven sequential path.
//
// Derivation (per head, chunk of L positions, incoming state S_in [P,N]):
// with inclusive cumulative decay P_i = Π_{j≤i} dA_j (P_{-1}=1),
//
//	S_i   = P_i·S_in + Σ_{m≤i} (P_i/P_m)·dt_m·(x_m⊗B_m)
//	y_i[p]= P_i·(S_in·C_i)[p] + Σ_{m≤i} (P_i/P_m)·dt_m·x_m[p]·(B_m·C_i) + D·x_i[p]
//	S_out = P_{L-1}·S_in + Σ_m (P_{L-1}/P_m)·dt_m·(x_m⊗B_m)
//
// dA_j = exp(dt_j·A) ∈ (0,1] (A<0), so P is non-increasing and every P_i/P_m ≤ 1 —
// numerically stable.

// mamba2Chunked runs the mixer over a whole sequence from a fresh state using the
// chunked scan, returning the same [hidden] output per position as mamba2Seq. chunk
// is the scan block size (chunk ≤ 0 ⇒ whole sequence; chunk == 1 reduces exactly to
// the sequential step).
func mamba2Chunked(h [][]float32, w *mamba2Weights, p mamba2Params, eps float64, chunk int) [][]float32 {
	n := len(h)
	if n == 0 {
		return nil
	}
	dInner, convDim, K := p.dInner(), p.convDim(), p.DConv
	N, P := p.DState, p.HeadDim
	gSize := p.NGroups * N
	repeat := p.NHeads / p.NGroups
	if chunk <= 0 || chunk > n {
		chunk = n
	}

	// --- position-local stages (identical arithmetic to mamba2Step) ---
	z := make([][]float32, n)   // gate, per position [dInner]
	x := make([][]float32, n)   // conv-x, per position [dInner]
	Bs := make([][]float32, n)  // conv-B, per position [gSize]
	Cs := make([][]float32, n)  // conv-C, per position [gSize]
	dtH := make([][]float32, n) // softplus(dt+bias), per position [nHeads]
	xBC := make([][]float32, n) // pre-conv xBC, per position [convDim]
	for t := range n {
		proj := matvec(w.inProj, p.projDim(), p.Hidden, h[t])
		z[t] = append([]float32(nil), proj[:dInner]...)
		xBC[t] = append([]float32(nil), proj[dInner:dInner+convDim]...)
		dtRaw := proj[dInner+convDim:]
		d := make([]float32, p.NHeads)
		for head := range p.NHeads {
			d[head] = softplusf(dtRaw[head] + w.dtBias[head])
		}
		dtH[t] = d
	}
	// Depthwise causal conv over xBC (+bias, +SiLU), then split into x/B/C.
	for t := range n {
		conv := make([]float32, convDim)
		for c := range convDim {
			s := w.convB[c] + w.convW[c*K+(K-1)]*xBC[t][c]
			for j := 0; j < K-1; j++ {
				if tt := t - (K - 1) + j; tt >= 0 {
					s += w.convW[c*K+j] * xBC[tt][c]
				}
			}
			conv[c] = silu(s)
		}
		x[t] = conv[:dInner]
		Bs[t] = conv[dInner : dInner+gSize]
		Cs[t] = conv[dInner+gSize : dInner+2*gSize]
	}

	// --- the chunked recurrence (the actual scan), per head ---
	y := make([][]float32, n)
	for t := range n {
		y[t] = make([]float32, dInner)
	}
	for head := range p.NHeads {
		grp := head / repeat
		A := -math.Exp(float64(w.aLog[head]))
		Dh := w.d[head]
		S := make([]float32, P*N) // [P,N], persists across chunks
		for start := 0; start < n; start += chunk {
			end := min(start+chunk, n)
			L := end - start
			// inclusive cumulative decay within the chunk
			Pc := make([]float32, L)
			acc := float32(1)
			for i := range L {
				da := float32(math.Exp(float64(dtH[start+i][head]) * A))
				acc *= da
				Pc[i] = acc
			}
			for i := range L {
				t := start + i
				Ci := Cs[t][grp*N : grp*N+N]
				out := y[t][head*P : head*P+P]
				xi := x[t][head*P : head*P+P]
				// term1: P_i·(S_in·C_i) + D·x_i
				for pi := range P {
					var s float32
					row := S[pi*N : pi*N+N]
					for nn := range N {
						s += row[nn] * Ci[nn]
					}
					out[pi] = Pc[i]*s + Dh*xi[pi]
				}
				// term2: Σ_{m≤i} (P_i/P_m)·dt_m·x_m[p]·(B_m·C_i)
				for m := 0; m <= i; m++ {
					Bm := Bs[start+m][grp*N : grp*N+N]
					var bc float32
					for nn := range N {
						bc += Bm[nn] * Ci[nn]
					}
					coef := (Pc[i] / Pc[m]) * dtH[start+m][head] * bc
					xm := x[start+m][head*P : head*P+P]
					for pi := range P {
						out[pi] += coef * xm[pi]
					}
				}
			}
			// S_out = P_{L-1}·S_in + Σ_m (P_{L-1}/P_m)·dt_m·(x_m⊗B_m)
			cl := Pc[L-1]
			for i := range S {
				S[i] *= cl
			}
			for m := range L {
				Bm := Bs[start+m][grp*N : grp*N+N]
				xm := x[start+m][head*P : head*P+P]
				sc := (cl / Pc[m]) * dtH[start+m][head]
				for pi := range P {
					xc := sc * xm[pi]
					row := S[pi*N : pi*N+N]
					for nn := range N {
						row[nn] += xc * Bm[nn]
					}
				}
			}
		}
	}

	// --- grouped gated RMSNorm (y·SiLU(z)) then out_proj, per position ---
	out := make([][]float32, n)
	for t := range n {
		gated := make([]float32, dInner)
		for i := range dInner {
			gated[i] = y[t][i] * silu(z[t][i])
		}
		gatedRMSNormGrouped(gated, w.normW, p.NormGroups, eps)
		out[t] = matvec(w.outProj, p.Hidden, dInner, gated)
	}
	return out
}

package decoder

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
)

// P26c (docs/queue-performance.md). The batched Qwen3.5 forward (runLayersQwen35N) runs every projection as one matmul over the prompt, but then called
// deltaNetCore once per token on ONE thread: on Qwen3.5-0.8B at int4 that was ~6.8 s of an 11 s, 684-token prefill (a CPU profile: deltaNetRecurrence
// ~4.9 s, the conv's SiLU ~2 s), while the matmuls it sits between ran on all 16 threads in about 2 s.
//
// deltaNetCoreN is deltaNetCore over K consecutive rows with the independent parts fanned out, and NOTHING reordered inside any one element:
//   - the depthwise conv (+SiLU) of row i reads only the mixed inputs of rows i-K+1..i, all known up front, so rows run in parallel;
//   - the recurrence is sequential over TOKENS but independent across value HEADS (each owns its [head_k_dim, head_v_dim] block of the state), so heads run in
//     parallel, each walking its K tokens in order with the very loop body deltaNetRecurrence runs;
//   - the gated RMSNorm is per token and per head, so rows run in parallel.
//
// Every float operation of every output element happens in the same order as in the per-token loop, so the result is BIT-IDENTICAL to it, state included
// (TestDeltaNetCoreN_bitIdentical holds it to exact bit equality). It is used only when no capture hook or timing is on, which keep the per-token loop.

// deltaNetParallel is the A/B handle and test seam for the fan-out, never an environment read.
var deltaNetParallel = true

// deltaNetMinWork is the smallest row count worth a fork/join; a variable only so a test can force the fan-out onto a tiny fixture.
var deltaNetMinWork = 8

// deltaNetFanouts counts layers that actually forked: a diagnostic for the tests' non-vacuity check, never branched on.
var deltaNetFanouts atomic.Int64

// deltaNetWorkers is the fan-out width over n independent items: GOMAXPROCS capped by linalg's process-wide width, and by n.
func deltaNetWorkers(n int) int {
	g := runtime.GOMAXPROCS(0)
	if w := linalg.ParallelWidth(); w > 0 && w < g {
		g = w
	}
	return max(1, min(g, n))
}

// fanOut runs fn over [0,n) split into contiguous chunks, one goroutine per chunk beyond the first (which runs on the caller).
func fanOut(n, workers int, fn func(lo, hi int)) {
	if workers <= 1 || n <= 1 {
		fn(0, n)
		return
	}
	per := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for i := 1; i < workers; i++ {
		lo, hi := i*per, min((i+1)*per, n)
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn(lo, hi)
		}()
	}
	fn(0, min(per, n))
	wg.Wait()
}

// deltaNetCoreNOK reports whether deltaNetCoreN applies; false keeps the per-token loop (the capture hook and the timing counters are per-token).
func deltaNetCoreNOK(K int) bool {
	return deltaNetParallel && K >= 2 && deltaCapHook == nil && !deltaNetTiming
}

// deltaNetCoreN is deltaNetCore over K rows. core [K*valueDim] receives the result; mixed [K*convDim], z [K*valueDim] and the raw gate projections bt, at
// [K*nv] are K-row slabs. mixed's rows are retained in st's conv window, as deltaNetCore retains its argument, so the caller must not reuse the slab.
func deltaNetCoreN(core, mixed, bt, at, z []float32, K int, w *deltaNetWeights, p qwen35Params, eps float64, st *deltaState) {
	nk, nv := p.NumKeyHeads, p.NumValueHeads
	hk, hv := p.KeyHeadDim, p.ValueHeadDim
	keyDim, valueDim := hk*nk, hv*nv
	convDim := 2*keyDim + valueDim
	Kc := p.ConvKernel
	rep := nv / nk
	qScale := float32(1 / math.Sqrt(float64(hk)))
	onEps := eps
	if p.ONormEps != 0 {
		onEps = p.ONormEps
	}
	workers := 1
	if K >= deltaNetMinWork {
		workers = deltaNetWorkers(K)
	}
	if workers > 1 {
		deltaNetFanouts.Add(1)
	}

	// Phase 1: conv + SiLU, and the L2-normalised q and k, per row. The window before row i is the last Kc-1 vectors of (prior ++ mixed rows 0..i-1).
	prior := st.convWin
	conv := make([]float32, K*convDim)
	qAll := make([]float32, K*keyDim)
	kAll := make([]float32, K*keyDim)
	fanOut(K, workers, func(lo, hi int) {
		wv := make([][]float32, Kc-1)
		for i := lo; i < hi; i++ {
			seqLen := len(prior) + i
			lw := min(Kc-1, seqLen)
			for j := range Kc - 1 {
				idx := lw - (Kc - 1) + j
				wv[j] = nil
				if idx >= 0 {
					pos := seqLen - lw + idx // position in the virtual sequence prior ++ mixed rows
					if pos < len(prior) {
						wv[j] = prior[pos]
					} else {
						r := pos - len(prior)
						wv[j] = mixed[r*convDim : (r+1)*convDim]
					}
				}
			}
			cur := mixed[i*convDim : (i+1)*convDim]
			out := conv[i*convDim : (i+1)*convDim]
			for c := range convDim {
				s := w.convW[c*Kc+(Kc-1)] * cur[c] // j=Kc-1 tap = current token
				for j := 0; j < Kc-1; j++ {
					if wv[j] != nil {
						s += w.convW[c*Kc+j] * wv[j][c]
					}
				}
				out[c] = silu(s)
			}
			for headK := 0; headK < nk; headK++ {
				l2normScaledInto(qAll[i*keyDim+headK*hk:i*keyDim+(headK+1)*hk], out[headK*hk:(headK+1)*hk], qScale)
				l2normScaledInto(kAll[i*keyDim+headK*hk:i*keyDim+(headK+1)*hk], out[keyDim+headK*hk:keyDim+(headK+1)*hk], 1)
			}
		}
	})

	// Phase 2: the gated delta-rule recurrence. Heads in parallel; each head walks the K tokens in order, with deltaNetRecurrence's own loop body.
	clear(core[:K*valueDim])
	fanOut(nv, min(workers, nv), func(lo, hi int) {
		kv := make([]float32, hv)
		for headV := lo; headV < hi; headV++ {
			headK := headV / rep
			S := st.s[headV*hk*hv : (headV+1)*hk*hv] // [hk, hv]
			sLen := len(S)
			for i := range K {
				q := qAll[i*keyDim+headK*hk : i*keyDim+(headK+1)*hk]
				k := kAll[i*keyDim+headK*hk : i*keyDim+(headK+1)*hk]
				v := conv[i*convDim+2*keyDim+headV*hv : i*convDim+2*keyDim+headV*hv+hv]
				g := w.negExpA[headV] * softplusf(at[i*nv+headV]+w.dtBias[headV]) // log-decay (negExpA = −exp(A_log))
				gt := float32(math.Exp(float64(g)))
				beta := sigmoidf(bt[i*nv+headV])
				if p.NegEigval {
					beta *= 2
				}
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
				out := core[i*valueDim+headV*hv : i*valueDim+headV*hv+hv]
				clear(kv)
				for kd := range hk {
					addScaled(kv, S[kd*hv:kd*hv+hv], k[kd])
				}
				delta := kv
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
				for kd := range hk {
					row := S[kd*hv : kd*hv+hv]
					kk, qq := k[kd], q[kd]
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
			}
		}
	})

	// Phase 3: gated RMSNorm (over head_v_dim, × SiLU(z)), per row and head.
	fanOut(K, workers, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			for headV := range nv {
				seg := core[i*valueDim+headV*hv : i*valueDim+headV*hv+hv]
				zt := z[i*valueDim+headV*hv : i*valueDim+headV*hv+hv]
				var ss float64
				for _, x := range seg {
					ss += float64(x) * float64(x)
				}
				inv := float32(1 / math.Sqrt(ss/float64(hv)+onEps))
				for vd := range hv {
					seg[vd] = seg[vd] * inv * w.normW[vd] * silu(zt[vd])
				}
			}
		}
	})

	// The conv window afterwards: the last Kc-1 of (prior ++ mixed rows), as the per-token loop leaves it (its rows are slices of the caller's slab).
	tail := make([][]float32, 0, Kc-1)
	total := len(prior) + K
	for pos := max(0, total-(Kc-1)); pos < total; pos++ {
		if pos < len(prior) {
			tail = append(tail, prior[pos])
		} else {
			r := pos - len(prior)
			tail = append(tail, mixed[r*convDim:(r+1)*convDim])
		}
	}
	st.convWin = tail
}

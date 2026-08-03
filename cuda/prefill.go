//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// prefillProf accumulates PrefillLast's per-category GPU time (test-only). The boundaries are stream
// syncs, so the category sum slightly exceeds the pipelined wall time (lost launch overlap) — it
// attributes where the time goes, not the fully-overlapped total.
type prefillProf struct {
	gemv, attn, glue time.Duration
}

type profCat int

const (
	gemvCat profCat = iota
	attnCat
	glueCat
)

// profTic syncs the stream and returns a start time when profiling is on; a no-op zero Time otherwise.
func (r *cudaResident) profTic() time.Time {
	if r.prof == nil {
		return time.Time{}
	}
	_ = r.stream.Sync()
	return time.Now()
}

// profToc syncs the stream and adds the elapsed time to the named category (no-op when profiling off).
func (r *cudaResident) profToc(cat profCat, t0 time.Time) {
	if r.prof == nil {
		return
	}
	_ = r.stream.Sync()
	d := time.Since(t0)
	switch cat {
	case gemvCat:
		r.prof.gemv += d
	case attnCat:
		r.prof.attn += d
	case glueCat:
		r.prof.glue += d
	}
}

// PrefillLast (decoder.Prefiller) ingests a whole prompt in ONE weight-stationary pass and returns the
// logits for the last token — the batched (M=len) counterpart of the sequential ForwardNoLogits loop.
// It fixes the ~128-token Ollama crossover: goinfer's sequential prefill reads every weight once PER
// PROMPT TOKEN (weight-bandwidth-bound at ~6 ms/token), while this reads each weight once for all M
// tokens (the gemv_w4a8_batched amortization). The K/V it writes is BIT-IDENTICAL to the sequential
// path row-for-row (every batched kernel is the M=1 kernel with an M dimension, per-row math verbatim),
// so decode from the last prompt token stays byte-identical.
//
// It handles the plain dense unfused forward only (Llama/Qwen2/Mistral-class): rmsnorm→Q/K/V→rope+kv→
// causal windowed attention→o-proj→rmsnorm→gate/up→swiglu→down. Anything it does not cover — MoE, the
// Gemma parallel dense‖MoE, sandwich norms, per-head QK-norm, K=V (Gemma), int8 weights, non-uniform
// per-layer geometry, or a prompt past the KV cap — returns an error so decoder/model.go falls back to
// the sequential KV-only prefill (which is correct for every family). Uniform-only is enforced against
// layer 0; a non-uniform family trips the guard and declines rather than reading a wrong stride.
func (r *cudaResident) PrefillLast(embeddings [][]float32, startPos int) ([]float32, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	if !r.prefillReady {
		return nil, fmt.Errorf("cuda prefill: batched kernels unavailable")
	}
	if r.moe || r.gemma4Moe || r.sandwich || r.qkNorm {
		return nil, fmt.Errorf("cuda prefill: arch needs the sequential path (moe/gemma4moe/sandwich/qknorm)")
	}
	if e := r.checkCap(startPos, M); e != nil {
		return nil, e
	}
	L0 := &r.layers[0]
	hd, nKV, qDim, kvDim, rhalf := L0.hd, L0.nKV, L0.qDim, L0.kvDim, L0.rhalf
	// Uniform geometry + all-int4 + no K=V, checked across every layer (decline, never mis-stride).
	for l := range r.layers {
		Ly := &r.layers[l]
		if Ly.hd != hd || Ly.nKV != nKV || Ly.qDim != qDim || Ly.kvDim != kvDim || Ly.rhalf != rhalf {
			return nil, fmt.Errorf("cuda prefill: non-uniform layer geometry at %d", l)
		}
		if Ly.kEqV {
			return nil, fmt.Errorf("cuda prefill: K=V layer at %d needs the sequential path", l)
		}
		if Ly.q.kind != "int4" || Ly.k.kind != "int4" || Ly.v.kind != "int4" ||
			Ly.o.kind != "int4" || Ly.g.kind != "int4" || Ly.u.kind != "int4" || Ly.d.kind != "int4" {
			return nil, fmt.Errorf("cuda prefill: non-int4 weight at layer %d needs the sequential path", l)
		}
	}
	hidden, inter := r.hidden, r.inter

	var out []float32
	err := r.do(func() error {
		// --- M-sized scratch (device), freed at the end. Allocation panics on OOM (recovered by the
		// caller's guard → declines to the sequential path), so a too-long prompt never proceeds half-set.
		xB := r.af(M * hidden)
		aqB, aScB := r.ai(M*hidden/4), r.af(M)
		qBb, kBb, vBb := r.af(M*qDim), r.af(M*kvDim), r.af(M*kvDim)
		cctxB := r.af(M * qDim)
		cqB, cScB := r.ai(M*qDim/4), r.af(M)
		mqB, mScB := r.ai(M*hidden/4), r.af(M)
		gOb, uOb := r.af(M*inter), r.af(M*inter)
		dqB, dScB, dScrB := r.ai(M*inter/4), r.af(M), r.af(M*inter)
		scratch := []Buffer{xB, aqB, aScB, qBb, kBb, vBb, cctxB, cqB, cScB, mqB, mScB, gOb, uOb, dqB, dScB, dScrB}
		defer func() {
			for _, b := range scratch {
				r.dev.ReleaseBuf(b)
			}
		}()

		// Upload the M embeddings contiguously as xB[M, hidden] (already FeatEmbedScale-scaled by the
		// caller, exactly as Forward/ForwardNoLogits receive them).
		xhost := make([]float32, M*hidden)
		for m, e := range embeddings {
			copy(xhost[m*hidden:(m+1)*hidden], e)
		}
		if e := gpu.Upload(xB, xhost); e != nil {
			return e
		}

		ropeN := r.nH*rhalf + nKV*rhalf + nKV*(hd-2*rhalf)
		for l := 0; l < r.nLayers; l++ {
			Ly := &r.layers[l]
			qb, kb, vb := ArgNull(), ArgNull(), ArgNull()
			if Ly.hasBias {
				qb, kb, vb = Arg(Ly.qb), Arg(Ly.kb), Arg(Ly.vb)
			}
			// segA: rmsnorm+quant (glue), then Q/K/V GEMVs. Category timers (r.prof) sync r.stream at
			// each group boundary; nil in production, so the launch sequence is otherwise unchanged.
			t := r.profTic()
			if e := r.bRmsB(xB, Ly.preNorm, hidden, aqB, aScB, M); e != nil {
				return e
			}
			r.profToc(glueCat, t)
			t = r.profTic()
			if e := r.bGemvB(Ly.q, aqB, aScB, qb, qBb, M, 0); e != nil {
				return e
			}
			if e := r.bGemvB(Ly.k, aqB, aScB, kb, kBb, M, 0); e != nil {
				return e
			}
			if e := r.bGemvB(Ly.v, aqB, aScB, vb, vBb, M, 0); e != nil {
				return e
			}
			r.profToc(gemvCat, t)
			// rope + kv-store (glue): token m at absolute position startPos+m; rotates q/k, writes K/V.
			t = r.profTic()
			if e := r.launch(r.bRopeKV, LaunchConfig{GridX: uint32((ropeN + 255) / 256), GridY: uint32(M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
				Arg(qBb), Arg(kBb), Arg(vBb), Arg(Ly.invF), Arg(r.kc[l]), Arg(r.vc[l]),
				gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
				gpu.ArgValue(int32(startPos)), gpu.ArgValue(int32(rhalf)), gpu.ArgValue(int32(M))); e != nil {
				return e
			}
			r.profToc(glueCat, t)
			// causal + per-row sliding-window attention; block 128 matches the M=1 attention reduce.
			maxNWin := startPos + M
			if Ly.window > 0 && int(Ly.window) < maxNWin {
				maxNWin = int(Ly.window)
			}
			t = r.profTic()
			if e := r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1,
				SharedMemBytes: uint32((maxNWin + 128) * 4)},
				Arg(qBb), Arg(r.kc[l]), Arg(r.vc[l]), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)),
				gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(startPos)), gpu.ArgValue(r.attnScale),
				gpu.ArgValue(Ly.window), gpu.ArgValue(int32(M)), Arg(cctxB)); e != nil {
				return e
			}
			r.profToc(attnCat, t)
			// segB: ctx-quant (glue), o-proj (gemv, accum into residual), MLP.
			t = r.profTic()
			if e := r.launch(r.bQuant, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
				Arg(cctxB), gpu.ArgValue(int32(qDim)), Arg(cqB), Arg(cScB), gpu.ArgValue(int32(M))); e != nil {
				return e
			}
			r.profToc(glueCat, t)
			t = r.profTic()
			if e := r.bGemvB(Ly.o, cqB, cScB, ArgNull(), xB, M, 1); e != nil {
				return e
			}
			r.profToc(gemvCat, t)
			t = r.profTic()
			if e := r.bRmsB(xB, Ly.postNorm, hidden, mqB, mScB, M); e != nil {
				return e
			}
			r.profToc(glueCat, t)
			t = r.profTic()
			if e := r.bGemvB(Ly.g, mqB, mScB, ArgNull(), gOb, M, 0); e != nil {
				return e
			}
			if e := r.bGemvB(Ly.u, mqB, mScB, ArgNull(), uOb, M, 0); e != nil {
				return e
			}
			r.profToc(gemvCat, t)
			t = r.profTic()
			if e := r.launch(r.bSw, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
				Arg(gOb), Arg(uOb), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(0)), gpu.ArgValue(int32(inter)),
				gpu.ArgValue(r.act), Arg(dqB), Arg(dScB), Arg(dScrB), gpu.ArgValue(int32(M))); e != nil {
				return e
			}
			r.profToc(glueCat, t)
			t = r.profTic()
			if e := r.bGemvB(Ly.d, dqB, dScB, ArgNull(), xB, M, 1); e != nil {
				return e
			}
			r.profToc(gemvCat, t)
		}

		// Final norm + LM head on the LAST row only — copy xB[M-1] into the M=1 scratch and reuse the
		// exact Forward tail, so the returned logits are bit-identical to a sequential Forward at the
		// last position (given identical residual, which the KV/logits gate checks). Drain the layer
		// launches first: they run on r.stream, and the DtoH below is not ordered after that stream.
		if e := r.stream.Sync(); e != nil {
			return e
		}
		if e := gpu.Download(xB, xhost); e != nil {
			return e
		}
		if e := gpu.Upload(r.x, xhost[(M-1)*hidden:]); e != nil {
			return e
		}
		if e := r.rms(r.x, r.finalNorm, r.aq, r.aSc); e != nil {
			return e
		}
		if e := r.doG(r.lmW, r.aq, r.aSc, ArgNull(), r.logits, 0); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		if e := gpu.ReadToHost(r.logits, r.logitsPinned); e != nil {
			return e
		}
		out = append([]float32(nil), r.logitsHost...)
		return r.launchErr
	})
	if err != nil {
		return nil, err
	}
	// Final-logit softcap (Gemma) — host-side, exactly as step(). No-op (0) for the dense families
	// this path serves, but kept so the contract matches Forward if a softcapped dense arch appears.
	if r.finalSoftcap > 0 {
		sc := r.finalSoftcap
		for j, v := range out {
			out[j] = sc * float32(math.Tanh(float64(v/sc)))
		}
	}
	return out, nil
}

// bRmsB launches rmsnorm_quant_batched over M rows (shared = [blockDim]+[hidden]).
func (r *cudaResident) bRmsB(x, w Buffer, N int, qOut, sOut Buffer, M int) error {
	return r.launch(r.bRms, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32((256 + N) * 4)},
		Arg(x), Arg(w), gpu.ArgValue(int32(N)), gpu.ArgValue(r.eps), gpu.ArgValue(r.addOneArg()),
		Arg(qOut), Arg(sOut))
}

// bGemvB launches gemv_w4a8_batched — same GridX/BlockX as the M=1 doG (the M loop is inside the
// kernel), so the per-output float accumulation order matches the sequential GEMV bit-for-bit.
func (r *cudaResident) bGemvB(wt cudaWQ, a, as Buffer, bias KernelArg, dst Buffer, M int, accum int32) error {
	if wt.kind != "int4" {
		return fmt.Errorf("cuda prefill: batched GEMV is int4-only, got %q", wt.kind)
	}
	cfg := LaunchConfig{GridX: uint32((wt.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
	return r.launch(r.bGemv, cfg, Arg(wt.W), Arg(a), Arg(wt.ws16), Arg(as), bias,
		gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/8)), gpu.ArgValue(int32(wt.K/32)),
		gpu.ArgValue(int32(M)), Arg(dst), gpu.ArgValue(accum))
}

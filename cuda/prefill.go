//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"strings"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
)

// errPrefillDeclined marks an ARCH/GEOMETRY decline from the batched path (MoE, K=V, non-int4,
// non-uniform geometry, or missing batched kernels) — as opposed to a real compute/cap error. The
// batched forward covers the plain dense unfused family only; callers that can fall back to the
// sequential per-token path (ForwardN for spec verify, model.go for prefill) test errors.Is(err,
// errPrefillDeclined) to distinguish "this arch can't batch, use the slow path" (recoverable) from
// "the batched kernels failed" (propagate). Wrapped into each decline so the message stays specific.
var errPrefillDeclined = errors.New("cuda prefill: batched path declined (arch/geometry)")

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
// PrefillLast ingests a whole prompt in one batched pass, returning the last token's logits.
func (r *cudaResident) PrefillLast(embeddings [][]float32, startPos int) ([]float32, error) {
	outs, _, err := r.prefillCore(embeddings, startPos, tailLastLogits)
	if err != nil {
		return nil, err
	}
	return outs[len(outs)-1], nil
}

// PrefillLastN is the D1 (speculative-decode) verify primitive: the SAME batched pass, but returns
// the logits at ALL M positions (row m = the target's prediction for position startPos+m+1). The
// batched layer stack is bit-identical to sequential per position (TestPrefillLast_e2e), and the
// final norm + LM head is applied per row exactly as PrefillLast applies it to the last — so each
// row's logits equal a sequential Forward's, which is what makes greedy accept lossless.
func (r *cudaResident) PrefillLastN(embeddings [][]float32, startPos int) ([][]float32, error) {
	outs, _, err := r.prefillCore(embeddings, startPos, tailAllLogits)
	return outs, err
}

// PrefillLastNArgmax is the spec-decode VERIFY primitive: the same batched pass, returning only
// each row's argmax token id — which is all the accept decision needs.
//
// It exists because PrefillLastN's per-row tail re-reads the LM head's weights ONCE PER ROW
// (~389 M params, ~195 MB at int4), measured at 1.046 ms marginal per row against a 0.934 ms
// single-row head — no amortization at all, in the one place the batched pass exists to provide
// it. This tail instead runs ONE batched final-norm, ONE batched head GEMV over all M rows, and M
// argmax reductions, then reads back 4 bytes per row instead of 608 KB.
//
// LOSSLESSNESS: the accept decision compares the drafted token against the target's argmax, so
// only the ARGMAX must match the sequential path — not the logits bit-for-bit. That is a strictly
// weaker requirement than PrefillLastN's, and it is what makes batching the head admissible at
// all. TestPrefillLastNArgmax_matchesPerRow gates it.
func (r *cudaResident) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	_, ids, err := r.prefillCore(embeddings, startPos, tailAllArgmax)
	return ids, err
}

// prefillStaticDecline reports why the batched path can't run for THIS model, or nil when it can. It
// covers exactly the MODEL-dependent guards — arch, per-layer geometry, weight kind, kernel
// availability — and deliberately not the prompt-dependent one (checkCap, which needs M/startPos).
// That split is what lets PrefillPath answer at LOAD time from the same code prefillCore enforces at
// call time: one source of truth, so the startup line can never drift from the actual decline.
//
// qk-norm (per-head Q/K RMSNorm) and Gemma sandwich norms are batched (qk_norm_batched /
// rmsnorm_f32_batched) — neither declines. backend.go asserts qNorm/kNorm length == headDim before
// residency (⇒ present when r.qkNorm), and the per-layer K=V / non-uniform / int4 checks still catch
// a Gemma-4-class layer this dense-batched path can't stride. MoE and the Gemma parallel dense‖MoE
// take the sequential path.
func (r *cudaResident) prefillStaticDecline() error {
	if !r.prefillReady {
		return fmt.Errorf("cuda prefill: batched kernels unavailable: %w", errPrefillDeclined)
	}
	if r.moe || r.gemma4Moe {
		return fmt.Errorf("cuda prefill: arch needs the sequential path (moe/gemma4moe): %w", errPrefillDeclined)
	}
	L0 := &r.layers[0]
	hd, nKV, qDim, kvDim, rhalf := L0.hd, L0.nKV, L0.qDim, L0.kvDim, L0.rhalf
	// Uniform geometry + all-int4 + no K=V, checked across every layer (decline, never mis-stride).
	for l := range r.layers {
		Ly := &r.layers[l]
		if Ly.hd != hd || Ly.nKV != nKV || Ly.qDim != qDim || Ly.kvDim != kvDim || Ly.rhalf != rhalf {
			return fmt.Errorf("cuda prefill: non-uniform layer geometry at %d: %w", l, errPrefillDeclined)
		}
		if Ly.kEqV {
			return fmt.Errorf("cuda prefill: K=V layer at %d needs the sequential path: %w", l, errPrefillDeclined)
		}
		if k := nonBatchableKind(Ly); k != "" {
			return fmt.Errorf("cuda prefill: %s weight at layer %d needs the sequential path: %w", k, l, errPrefillDeclined)
		}
	}
	return nil
}

// nonBatchableKind returns the weight kind of the layer's first projection the batched GEMVs can't
// handle, or "" when every projection is int4 or int8. The batched path dispatches per projection
// (bGemvB): int4 → gemv_w4a8_batched/_rn (group-scaled float accumulate), int8 → gemv_w8a8_batched
// (exact int32, §C6). A uniformly-int4, uniformly-int8, or MIXED int4mix bundle all batch; anything
// else (e.g. a native/f32 projection) declines. Naming the kind turns "declined" into an actionable
// startup message.
func nonBatchableKind(Ly *cudaLayer) string {
	for _, w := range []cudaWQ{Ly.q, Ly.k, Ly.v, Ly.o, Ly.g, Ly.u, Ly.d} {
		if w.kind != "int4" && w.kind != "int8" {
			return w.kind
		}
	}
	return ""
}

// PrefillPath (decoder.PrefillPathReporter) answers, at load, whether this model will get the batched
// prefill — before a single request has been served. Batched prefill now covers int4 AND int8 bundles
// (§C6); it still declines a native/f32 projection or a non-uniform/K=V geometry. Before int8 batched
// prefill landed, a dense model loaded at int8int8 built a fully resident decode path (looked healthy,
// decoded at 0.7× int4) but every prompt took the sequential per-token prefill — measured 1.73 s vs
// 0.19 s on a 300-token prompt (9×), 20× the CPU (4.56 vs 0.22 CPU-s), no compute hotspot: the executor
// spin-waiting through 300 sequential launches instead of one pass.
func (r *cudaResident) PrefillPath() (bool, string) {
	err := r.prefillStaticDecline()
	if err == nil {
		return true, "batched (one weight-stationary CUDA pass)"
	}
	// Detail without the wrapped sentinel, which says nothing a user can act on.
	detail := strings.TrimPrefix(strings.TrimSuffix(err.Error(), ": "+errPrefillDeclined.Error()), "cuda prefill: ")
	if len(r.layers) > 0 {
		if k := nonBatchableKind(&r.layers[0]); k != "" {
			return false, fmt.Sprintf("sequential — batched prefill needs int4 or int8 projections (%s weights) — ~9× slower TTFT, 20× CPU on a 300-token prompt", k)
		}
	}
	return false, "sequential — " + detail + " (slower TTFT: one forward per prompt token)"
}

// prefillCore runs the batched (M=len) forward. allLogits=false heads only the last row (PrefillLast);
// allLogits=true heads every row (PrefillLastN, spec-decode verify).
// prefill tail modes: what the batched pass does after the layer stack.
const (
	tailLastLogits = iota // head the LAST row only (PrefillLast)
	tailAllLogits         // head every row, per-row loop, full logits (PrefillLastN)
	tailAllArgmax         // batched head over all rows, return argmax ids (PrefillLastNArgmax)
)

func (r *cudaResident) prefillCore(embeddings [][]float32, startPos int, tail int) ([][]float32, []int, error) {
	M := len(embeddings)
	if M == 0 {
		return nil, nil, fmt.Errorf("cuda prefill: empty prompt")
	}
	if e := r.prefillStaticDecline(); e != nil {
		return nil, nil, e
	}
	if e := r.checkCap(startPos, M); e != nil {
		return nil, nil, e
	}
	L0 := &r.layers[0]
	hd, nKV, qDim, kvDim, rhalf := L0.hd, L0.nKV, L0.qDim, L0.kvDim, L0.rhalf
	hidden, inter := r.hidden, r.inter

	var outs [][]float32
	var ids []int
	err := r.do(func() error {
		r.launchErr = nil // N-04: clear the sticky accumulator first (like launchToken), so a prior
		// decode's discarded launch error isn't re-reported by this prefill.
		// --- M-sized scratch (device), freed at the end.
		//
		// The free list and its defer are registered BEFORE the first allocation, and each buffer
		// joins the list as it is created (audit C-24). Allocation PANICS on OOM per
		// gpu.NewBufferLenOf's contract, and at M=3000 this is hundreds of MB, so a partial
		// allocation is the expected failure on a nearly-full card — not a rare one. Building the
		// list first and deferring after (the previous shape) freed nothing at all when allocation
		// #10 of 17 panicked, because the defer had not been registered yet. That leaked only
		// because the panic used to kill the process anyway; now that runJob recovers it into a
		// decline, the leak would be real, repeatable, and would push the NEXT prompt closer to OOM.
		var scratch []Buffer
		defer func() {
			for _, b := range scratch {
				r.dev.ReleaseBuf(b)
			}
		}()
		af := func(n int) Buffer { b := r.af(n); scratch = append(scratch, b); return b }
		ai := func(n int) Buffer { b := r.ai(n); scratch = append(scratch, b); return b }

		xB := af(M * hidden)
		aqB, aScB := ai(M*hidden/4), af(M)
		qBb, kBb, vBb := af(M*qDim), af(M*kvDim), af(M*kvDim)
		cctxB := af(M * qDim)
		cqB, cScB := ai(M*qDim/4), af(M)
		mqB, mScB := ai(M*hidden/4), af(M)
		gOb, uOb := af(M*inter), af(M*inter)
		dqB, dScB, dScrB := ai(M*inter/4), af(M), af(M*inter)
		// Sandwich families (Gemma) norm the attention / MLP sublayer output BEFORE adding it to the
		// residual, so the o-proj and down GEMVs write a temp instead of accumulating in place. One
		// [M, hidden] buffer, reused for both (the two uses are sequential).
		var sbB Buffer
		if r.sandwich {
			sbB = af(M * hidden)
		}
		residMN := uint32((M*hidden + 255) / 256)

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
			// per-head Q/K RMSNorm BEFORE rope (Qwen3): in place on qBb/kBb, one block per (head,token).
			// Bit-identical to the decode qk_norm applied per token (same f64 reduction, same addOne).
			if r.qkNorm {
				t = r.profTic()
				addOne := int32(0)
				if r.rmsAddOne {
					addOne = 1
				}
				if e := r.launch(r.bQKN, LaunchConfig{GridX: uint32(r.nH + nKV), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 8},
					Arg(qBb), Arg(kBb), Arg(Ly.qNorm), Arg(Ly.kNorm),
					gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(hd)),
					gpu.ArgValue(r.eps), gpu.ArgValue(addOne), gpu.ArgValue(int32(M))); e != nil {
					return e
				}
				r.profToc(glueCat, t)
			}
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
			if r.sandwich {
				// o-proj → temp (accum=0), Gemma post-attn RMSNorm per row, then add to residual.
				// Mirrors segB's decode sandwich path (o-proj → normF32 → residual) exactly, per row.
				if e := r.bGemvB(Ly.o, cqB, cScB, ArgNull(), sbB, M, 0); e != nil {
					return e
				}
				if e := r.bNormF32B(sbB, Ly.postAttnNorm, hidden, M); e != nil {
					return e
				}
				if e := r.launch(r.bRes, LaunchConfig{GridX: residMN, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
					Arg(xB), Arg(sbB), gpu.ArgValue(int32(M*hidden))); e != nil {
					return e
				}
			} else if e := r.bGemvB(Ly.o, cqB, cScB, ArgNull(), xB, M, 1); e != nil {
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
			if r.sandwich {
				// down → temp (accum=0), Gemma post-MLP RMSNorm per row, then add to residual.
				if e := r.bGemvB(Ly.d, dqB, dScB, ArgNull(), sbB, M, 0); e != nil {
					return e
				}
				if e := r.bNormF32B(sbB, Ly.postMLPNorm, hidden, M); e != nil {
					return e
				}
				if e := r.launch(r.bRes, LaunchConfig{GridX: residMN, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1},
					Arg(xB), Arg(sbB), gpu.ArgValue(int32(M*hidden))); e != nil {
					return e
				}
			} else if e := r.bGemvB(Ly.d, dqB, dScB, ArgNull(), xB, M, 1); e != nil {
				return e
			}
			r.profToc(gemvCat, t)
		}

		// Final norm + LM head, per row — copy xB[m] into the M=1 scratch and reuse the exact Forward
		// tail, so each row's logits are bit-identical to a sequential Forward at position startPos+m
		// (given identical residual, which the KV/logits gate checks). allLogits=false heads only the
		// last row (the crossover-fixing PrefillLast); allLogits=true heads every row (verify). Drain
		// the layer launches first: they run on r.stream, and the DtoH below is not ordered after it.
		if e := r.stream.Sync(); e != nil {
			return e
		}
		if e := gpu.Download(xB, xhost); e != nil {
			return e
		}
		if tail == tailAllArgmax {
			return r.batchedHeadArgmax(xB, aqB, aScB, M, &ids)
		}
		first := M - 1
		if tail == tailAllLogits {
			first = 0
		}
		outs = make([][]float32, M)
		for m := first; m < M; m++ {
			if e := gpu.Upload(r.x, xhost[m*hidden:(m+1)*hidden]); e != nil {
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
			outs[m] = append([]float32(nil), r.logitsHost...)
		}
		return r.launchErr
	})
	if err != nil {
		// An OOM inside the job arrives as a recovered panic (runJob, audit C-24) carrying aikit
		// MustBuf's "device allocation failed" message. For prefill specifically that is a DECLINE, not
		// a request failure: the sequential per-token path needs no M-sized scratch and will serve this
		// prompt. Match that OOM SENTINEL, not any "panicked" (audit R-20): a future programming-bug
		// panic in the batched path must surface as a real error, not be silently absorbed into the
		// ~9×-slower sequential path. Errors that are already declines (static guards, checkCap) keep
		// their own wrapping.
		if strings.Contains(err.Error(), "device allocation failed") && !errors.Is(err, errPrefillDeclined) {
			return nil, nil, fmt.Errorf("cuda prefill: out of device memory for M=%d scratch (%w): %v", M, errPrefillDeclined, err)
		}
		return nil, nil, err
	}
	// Final-logit softcap (Gemma) — host-side, exactly as step(). No-op (0) for the dense families
	// this path serves, but kept so the contract matches Forward if a softcapped dense arch appears.
	for _, out := range outs {
		applySoftcap(out, r.finalSoftcap)
	}
	return outs, ids, nil
}

// batchedHeadArgmax is tailAllArgmax's tail: ONE batched final-norm, ONE batched head GEMV over
// all M rows, M argmax reductions, then a 4-bytes-per-row readback.
//
// The win it exists for: the per-row tail issues the head as an M=1 GEMV per row, so the head's
// ~389 M parameters are re-read from VRAM M times. Measured, that marginal row costs 1.046 ms
// against a 0.934 ms single-row head — no amortization whatsoever, in the one place the batched
// pass exists to provide it.
//
// Buffers are allocated lazily HERE because af/ai need r.dev's context current, which holds on
// the executor thread this runs on. They are sized to M and reused; a wider block reallocates
// once. The old buffers are left to the device ledger rather than freed mid-job, which is the
// same lifetime the rest of the resident scratch has.
func (r *cudaResident) batchedHeadArgmax(xB, aqB, aScB Buffer, M int, out *[]int) error {
	if M > r.logitsBCap {
		r.logitsB = r.af(M * r.vocab)
		r.logitsBCap = M
	}
	// Batched final norm + quant over all M rows, exactly as the layer loop norms its input.
	if e := r.bRmsB(xB, r.finalNorm, r.hidden, aqB, aScB, M); e != nil {
		return e
	}
	// ONE head GEMV for all M rows: the weights are read once instead of M times.
	if e := r.bGemvB(r.lmW, aqB, aScB, ArgNull(), r.logitsB, M, 0); e != nil {
		return e
	}
	// The argmax is taken on the HOST, not by M launches of argmax_reduce over slices of the
	// batched logits. gocudrv exposes no buffer view or offset (the same limitation noted for the
	// SwiGLU halves at resident.go), so a per-row reduction would need either a batched argmax
	// kernel or M single-row buffers. Neither is worth it here: the win being chased is the M
	// weight READS of a 195 MB head, and a host argmax over M x vocab costs ~1.1 ms against the
	// ~6.4 ms that recovers. A batched argmax kernel is a later, separate ~1 ms.
	if e := r.stream.Sync(); e != nil {
		return e
	}
	host := make([]float32, M*r.vocab)
	if e := gpu.Download(r.logitsB, host); e != nil {
		return e
	}
	ids := make([]int, M)
	for m := 0; m < M; m++ {
		row := host[m*r.vocab : (m+1)*r.vocab]
		bi, bv := 0, row[0]
		for i, v := range row {
			if v > bv {
				bi, bv = i, v
			}
		}
		ids[m] = bi
	}
	*out = ids
	return r.launchErr
}

// bRmsB launches rmsnorm_quant_batched over M rows (shared = [blockDim]+[hidden]).
func (r *cudaResident) bRmsB(x, w Buffer, N int, qOut, sOut Buffer, M int) error {
	return r.launch(r.bRms, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1,
		SharedMemBytes: uint32((256 + N) * 4)},
		Arg(x), Arg(w), gpu.ArgValue(int32(N)), gpu.ArgValue(r.eps), gpu.ArgValue(r.addOneArg()),
		Arg(qOut), Arg(sOut))
}

// bNormF32B is the batched counterpart of normF32 (Gemma sandwich post-norm): a plain in-place
// f32 RMSNorm of an [M, H] sublayer output, one block per row (grid.y = m), blockDim 256 to match
// the decode reduction tree. No-op when the arch declares no sandwich norms (empty weight buffer).
func (r *cudaResident) bNormF32B(x, w Buffer, H, M int) error {
	if w.Len() == 0 {
		return nil
	}
	return r.launch(r.bNormF32, LaunchConfig{GridX: 1, GridY: uint32(M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(x), Arg(w), gpu.ArgValue(int32(H)), gpu.ArgValue(r.eps), gpu.ArgValue(r.addOneArg()), gpu.ArgValue(int32(M)))
}

// rnBlockRows must equal RN in gemv_w4a8_rn.cu — each warp computes this many output rows, so the grid
// covers ceil(N/rnBlockRows) warps. Bit-identical for any RN; 2 is the profiled knee (halves the L1TEX
// load count → halves the scoreboard stall, 4.41→3.38 ms, at the 64-reg / 100%-occupancy limit).
const rnBlockRows = 2

// bGemvB launches the register-blocked gemv_w4a8_rn (RN rows/warp). Bit-identical to the M=1 GEMV
// (each row keeps its own facc across all K, one warp-reduce), just with each activation load reused
// across RN rows. Grid = ceil(N/RN) warps → ceil(that/8) blocks of 256 threads.
func (r *cudaResident) bGemvB(wt cudaWQ, a, as Buffer, bias KernelArg, dst Buffer, M int, accum int32) error {
	switch wt.kind {
	case "int4":
		warps := (wt.N + rnBlockRows - 1) / rnBlockRows
		cfg := LaunchConfig{GridX: uint32((warps + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		return r.launch(r.bRN, cfg, Arg(wt.W), Arg(a), Arg(wt.ws16), Arg(as), bias,
			gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/8)), gpu.ArgValue(int32(wt.K/32)),
			gpu.ArgValue(int32(M)), Arg(dst), gpu.ArgValue(accum))
	case "int8":
		// Batched W8A8 (§C6). One warp per output row (8 warps/block), same layout as doG's int8
		// GEMV (wt.ws = per-row f32 scale, K/4 int words). Bit-identical to gemv_w8a8_fwd by
		// construction — exact int32 accumulation, so tiling M cannot change any element.
		cfg := LaunchConfig{GridX: uint32((wt.N + 7) / 8), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}
		return r.launch(r.bW8, cfg, Arg(wt.W), Arg(a), Arg(wt.ws), Arg(as), bias,
			gpu.ArgValue(int32(wt.N)), gpu.ArgValue(int32(wt.K/4)), gpu.ArgValue(int32(M)),
			Arg(dst), gpu.ArgValue(accum))
	default:
		return fmt.Errorf("cuda prefill: batched GEMV is int4/int8-only, got %q", wt.kind)
	}
}

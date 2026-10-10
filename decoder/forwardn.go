package decoder

import (
	"context"
	"fmt"
	"math"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
)

// cpuFastAttention reports whether the live environment enables f32 prefill attention (GOINFER_CPU_FAST_ATTENTION; on unless
// set to "0", which is what --cpu-exact-prefill sets). It is the reading for a hand-built caller with no Model; a loaded model
// reads its own snapshot through (*Model).cpuFastAttention. It is read only here: every consumer receives the answer as an
// explicit argument, so no path can pick it up by accident (see runLayersFromEmbedN).
//
// Enabling it gives up two of the exact (acc64) path's guarantees for the model: spec-decode verify == sequential greedy
// (structurally prevented from applying there) and decode == prefill. A long prompt can therefore produce a different response at
// temperature 0 than under the exact kernel; decode itself is untouched. It also gives up split-invariance: f32 reassociation makes
// a prompt's KV depend on how the prompt was chunked, so a Session (a warm prefix, then a divergent suffix) stops matching a
// one-shot generate over the same tokens. That divergence is accepted (operator decision). TestSessionFastAttnDivergence pins
// it; the equality itself is gated under the exact kernel by TestSessionNgramSpecParity.
//
// It is floored by prompt length (fastAttnMinPrompt). MoE is deliberately not excluded, and nothing here checks the arch.
// Evidence, measured figures and the decision record: docs/code-notes/decoder.md#cpuFastAttention.
func cpuFastAttention() bool { return (*knobSet)(nil).cpuFastAttention() }

// cpuFastAttention is the per-model form: off when this model was loaded with
// Options.ExactPrefill, else the env var's answer.
func (m *Model) cpuFastAttention() bool { return !m.exactPrefill && m.knobs.cpuFastAttention() }

// ExactPrefill reports whether this model was loaded with Options.ExactPrefill — prompt ingestion
// must take each backend's bit-exact path. Backends consult it next to their own env var.
func (m *Model) ExactPrefill() bool { return m.exactPrefill }

// fastAttnMinPrompt is the prompt length (K, the suffix being prefilled) below which f32 prefill attention is not used, even
// when enabled: attention is O(K*nKeys), so the win grows with K while the divergence does not, and a short prompt gets a
// different answer for almost nothing.
//
// 512 is a judgement, not a measured crossover (the smallest K measured with a win over ~15%); a sweep that finds one should move
// it and say so. It is deliberately not configurable: a knob would be another way for prefill numerics to vary between two runs
// of the same build. Measurements: docs/code-notes/decoder.md#fastAttnMinPrompt.
const fastAttnMinPrompt = 512

// attnHeadsParThreshold is the per-call work (K*nKeys, the QKᵀ/AV size driver) below which attendBatchedHeads runs its heads
// serially through pool[0] instead of fanning out: the same "small work stays serial" rule int4ParThreshold applies. It is 0, so
// the fan-out is never skipped on size. Campaign record: docs/completed/task-attention-decode-cost.md (move (a)).
const attnHeadsParThreshold = 0

// attnGroupedNEONSize is the query-head group size (NumHeads/NumKVHeads) the aikit grouped acc64 kernels
// (MatmulQKAcc64Group/MatmulAVAcc64Group) have a NEON port for. Any other ratio still runs correctly through the Go fallback but
// slower than separate per-head calls, so attendGroupedHeads is gated to exactly this size rather than trusting the kernel's own
// dispatch to save a mismatched caller.
const attnGroupedNEONSize = 6

// attnGroupedNEONBlock is the NEON grouped kernels' own block width (8 keys per QK block, 8 dims per AV block). runSplitAligned
// rounds each worker's slice up to a multiple of it so no slice falls through to the much slower plain-Go grouped kernel for a
// ragged remainder.
const attnGroupedNEONBlock = 8

// attnGroupedMinKeys is the number of attended keys below which the per-head path runs unchanged. 128 is the smallest depth
// either R13 record measured a win at: a conservative floor, not the cache probe's own crossover (the cache already dedups a GQA
// group's shared reads below roughly K=1024). Records: docs/measurements/r13-distinct-bytes-probe-2026-09-19.md and
// docs/measurements/r13-neon-kernel-ab-2026-09-20.md.
const attnGroupedMinKeys = 128

// knobSet.attnGrouped (knobs.go) reports whether the grouped-kernel decode path may run; on unless GOINFER_ATTN_GROUPED=0, which
// restores the per-head path. The path also needs attnGroupedKernels (a per-architecture default, on for arm64 only).

// attnGroupedRuns counts attendGroupedHeads/attendGroupedLayer dispatches. Bit-identity hides dispatch inertness (a grouped path
// that never runs passes every correctness gate), so a test asserts it nonzero above attnGroupedMinKeys and zero below;
// moeExpertMajorRuns (mlp.go) is the same idiom.
var attnGroupedRuns int64

// canBatchN reports whether the batched M=K path applies: K>1 on a family without its own sequential forward, with a gated MLP
// and no learned position embedding (GPT-2-style models take the sequential fallback). Sparse MoE batches its attention, but its
// FFN stays per row because the router picks different experts per token.
func (m *Model) canBatchN(K int) bool {
	a := m.w.arch
	// Every family with its own sequential forward is excluded, derived from the dispatch table (ownForward) rather than restated
	// here: a restated list once fell a family behind, and a 2-token prompt ran the dense attention stack over conv layers that load
	// no q/k/v/o.
	if _, own := a.ownForward(); own {
		return false
	}
	return K > 1 && m.w.Embed.Rows() != 0 && !a.NonGatedMLP && !a.LearnedPosEmbed
}

// hasRecurrentState reports whether this model carries state mutated in place per token (a conv window, an SSM state, a
// linear-attention state), which no positional rewind can restore. It is the arch-side view of KVCache.hasRecurrentState(), read
// from the dispatch table's Recurrent bit so it can be asked before any cache exists (the resident path has none).
//
// One predicate, read everywhere (specRollbackSafe, resident_reuse.go, generate_vl.go): a second hand-written family list is how
// a family gets missed at a consumer. A new state kind is added to the dispatch table once and every caller follows.
func (m *Model) hasRecurrentState() bool {
	// Nil-safe because residentReuseLen is reachable from a Model with no weights loaded (its unit test builds one). A model with no
	// arch has no family and so no recurrent state.
	if m == nil || m.w == nil || m.w.arch == nil {
		return false
	}
	f, own := m.w.arch.ownForward()
	return own && f.Recurrent
}

// specRollbackSafe reports whether speculative decode's rollback (KVCache.TruncateTo after a partial accept) correctly restores
// this model's state. It is true for softmax/GQA and MLA, whose state lives in the cache and is resliced. It is false for every
// family hasRecurrentState marks Recurrent: TruncateTo does not roll back their in-place state, so verifying a K-token block
// over-advances it and the next round decodes from it, a silent distribution bug rather than a crash. It is also false for
// windowed models (see below). The n-gram speculative entry points refuse unsafe models and the caller falls back to plain
// decode; no checkpoint-at-block-start/restore path exists.
func (m *Model) specRollbackSafe() bool {
	a := m.w.arch
	if m.hasRecurrentState() {
		return false
	}
	// A staged sliding-window cache stores local layers in physical rings. Once a ring wraps (context > window), a rollback of more
	// than one position cannot restore the evicted positions, so verify would read stale history and diverge. The resident
	// positional path is itself safe, but the speculative loops can use the staged ring even when m.resident != nil (and a resident
	// CAS loss falls back to staged mid-flight), so windowed models are refused for speculation unconditionally; the cost is plain
	// decode. That keeps the staged rollback sites (KVCache.TruncateTo) to ring-free caches, where it is always exact. Re-enabling
	// windowed speculation means consuming an exact-rollback bool at each rollback site.
	if a.SlidingWindow > 0 {
		return false
	}
	return true
}

// forwardLayersN runs the embedding + all transformer layers + final norm over
// the K tokens in ids — appended as the next K positions of cache — and returns
// the [K, HiddenDim] post-final-norm hidden states (the LM head is the caller's).
// The weight matmuls run at M=K, so each weight streams from memory once and is
// reused across the K rows (aikit's column-blocked W8A8 kernel); attention stays
// per-position and causal. Bit-identical to K sequential forwards. Assumes
// canBatchN(len(ids)) — callers check.
func (m *Model) forwardLayersN(reqCtx context.Context, ids []int, cache *KVCache, fastAttn bool) ([]float32, error) {
	return m.runLayersFromEmbedN(reqCtx, m.embedN(ids), cache, fastAttn)
}

// embedN returns the [K*HiddenDim] embedding rows for ids (the per-row token embed
// times any embedding scale) — the text input to the batched stack.
func (m *Model) embedN(ids []int) []float32 {
	arch := m.w.arch
	hidden := arch.HiddenDim
	h := make([]float32, len(ids)*hidden)
	for i, id := range ids {
		m.w.Embed.Row(id, h[i*hidden:i*hidden+hidden])
	}
	if arch.EmbedScale != 0 && arch.EmbedScale != 1 {
		s := float32(arch.EmbedScale)
		for i := range h {
			h[i] *= s
		}
	}
	return h
}

// runLayersFromEmbedN runs the batched layer stack over K pre-embedded rows
// h [K*HiddenDim] — the batched analog of the streaming runLayersFromEmbed, so a
// multimodal caller can inject projected vision embeddings at <image> positions
// (and SetImageBlocks for the bidirectional mask) before the forward. Text rows
// must already carry the embedding scale; injected vision rows are raw (HF's
// masked_scatter overwrites the scaled placeholder embed). Returns the [K,
// HiddenDim] post-final-norm hidden states (the LM head is the caller's).
func (m *Model) runLayersFromEmbedN(reqCtx context.Context, h []float32, cache *KVCache, fastAttn bool) ([]float32, error) {
	arch := m.w.arch
	be := m.be
	hidden, nKV, hd := arch.HiddenDim, arch.NumKVHeads, arch.HeadDim
	// maxQDim, not qDim: a family with per-layer query heads (Laguna) reuses q/ctx across every layer, so they are sized for the
	// widest and sliced per layer below. maxHeads is NumHeads everywhere else.
	maxQDim, kvDim, inter := arch.maxHeads()*hd, nKV*hd, arch.IntermediateDim
	K := len(h) / hidden
	startPos := cache.Pos()

	norm := make([]float32, K*hidden)
	q := make([]float32, K*maxQDim)
	k := make([]float32, K*kvDim)
	v := make([]float32, K*kvDim)
	ctx := make([]float32, K*maxQDim)
	att := make([]float32, K*hidden)
	gate := make([]float32, K*inter)
	up := make([]float32, K*inter)
	mlpOut := make([]float32, K*hidden)
	// Batched per-head attention scratch (reused across layers; nKeys = startPos+K
	// is the same for every layer in this sweep). See attendBatchedHeads.
	maxKeys := startPos + K
	// fastAttn is the caller's statement that this sweep may use f32 attention and so diverge from the exact path (see
	// cpuFastAttention). It is a parameter, not a global read, so the guard is structural: spec-decode verify runs through forwardN
	// and must keep acc64 or "verify == sequential greedy" silently stops holding, and a runtime check could not tell the two callers
	// apart. The exclusions are applied here, at the single point where the decision becomes arithmetic, not at each caller: three
	// call sites pass cpuFastAttention() and verify passes false, so a per-caller guard would be three chances to forget.
	//
	// The floor keys on K, the suffix length, not on attention work (K*nKeys), deliberately: a warm 2048+128 suffix (K=128) does not
	// take the fast path although it attends over the whole prefix. The divergence was measured at the shapes the floor admits, and
	// re-keying would extend an accepted output change to short suffixes whose divergence nobody has measured. Changing it means
	// measuring at that shape first.
	//
	// MoE is not excluded: its measured divergence was close to the dense case and never reached the output
	// (docs/code-notes/decoder.md#Model.runLayersFromEmbedN.fastAttn). That evidence is one family, Mellum2, whose sliding-attention
	// layers cap nKeys; a full-attention MoE is unmeasured, so do not read it as "MoE in general".
	if fastAttn && K < fastAttnMinPrompt {
		fastAttn = false
	}
	useAcc64 := !fastAttn

	// Prefill attention runs its heads in parallel: each worker owns its scratch slot and the nH query heads are independent
	// (disjoint ctx writes, no shared mutable state), so bit-identity holds (TestPrefillAttnPoolInvariance gates it). The pool is
	// budgeted, not simply maxAttnWorkers: a slot's scores buffer is K*nKeys floats, so prefillAttnWorkersK lets the worker count fall
	// toward serial on long prompts rather than the allocation growing without bound.
	//
	// useAcc64 and cache.treeMask are fixed for this whole call, so fusedOK in attendBatchedHeads is the same for every layer: the
	// promise newHeadWorkerPool's wantFused needs to safely skip vt/scores.
	wantFusedPool := !useAcc64 && cache.treeMask == nil && arch.AttnLogitSoftcap == 0 // the fused tile never sees a score to cap
	attnPool := newHeadWorkerPoolK(m.knobs, prefillAttnWorkersK(m.knobs, K, maxKeys, hd, arch.maxHeads()), K, maxKeys, hd, wantFusedPool)
	// f32 scratch for the assembled local window (ring history + new rows) AND for
	// dequantizing int8 layers into for the f32 attention; ≤ maxKeys rows wide.
	// Allocated when the model has ring layers or an int8 cache.
	var alk, alv []float32
	if cache.localAny || cache.quant == kvI8 {
		alk = make([]float32, maxKeys*kvDim)
		alv = make([]float32, maxKeys*kvDim)
	}
	// Batch the qkv and gate/up projections (shared activation) so a GPU backend runs each group as one submit (BatchTiled). The
	// Workspace threshold is set to DefaultDecodeParallelThreshold, not aikit's prefill-scale default: this ws also carries small-K
	// forwardN calls (spec verify, K<=8), whose matmuls would otherwise run serial while decode's identical-shape M=1 matmul fans out.
	// Large-K prefill matmuls clear both thresholds.
	var ws linalg.Workspace
	var hq w4a8Act // the K-row normed block, quantized once for q/k/v and once for gate/up (R-13); reused per layer
	ws.SetThreshold(DefaultDecodeParallelThreshold)
	var qkvOps [3]linalg.W8A8Op
	var guOps [2]linalg.W8A8Op
	// Laguna attention-gate scratch, grown on first use. Its width depends on the
	// gate's granularity (nH or nH*hd) AND on the layer's head count, so it is sized
	// per layer rather than up front. nil for every other family.
	var gbuf []float32

	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }

	if m.layerPager != nil {
		defer m.layerPager.finishLayers()
	}

	for l := 0; l < arch.NumLayers; l++ {
		// An abandoned client must not leave this loop running: prefill is where the time goes. The context is checked per layer, which is
		// free at this granularity but not instant: cancellation latency is bounded by one layer's work (TestPrefillCancelMidFlight logs
		// it). The tail to tighten, if it ever matters, is a per-head check inside attendBatchedHeads.
		if err := reqCtx.Err(); err != nil {
			return nil, err
		}
		if m.layerPager != nil {
			m.layerPager.enterLayer(l) // dense weight streaming (#4)
		}
		lw := &m.w.Layers[l]
		// Resolved per layer, not hoisted: NormPlacementLinear (Olmo Hybrid) can give
		// this layer a different placement than the rest of the model. The batched twin
		// of runLayersFromEmbed's own per-layer resolution.
		placement := arch.normPlacementAt(l)
		sandwich := placement == NormSandwich4
		postOnly := placement == NormPostOnly
		postNorm := sandwich || postOnly
		parallel := placement == NormParallel
		global := arch.isGlobalLayer(l)
		// This layer's query width. q/ctx are sliced to it so every row stride, RoPE
		// call and matmul below sees the layer's OWN head count — the batched twin of
		// what causalAttention does for K=1.
		nH := arch.headsAt(l)
		qDim := nH * hd
		q, ctx := q[:K*qDim], ctx[:K*qDim]

		if !postOnly {
			for i := range K {
				normalizeInto(arch, row(norm, i, hidden), row(h, i, hidden), lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
			}
		} else {
			copy(norm, h)
		}
		if isW8A8(&lw.QProj) && isW8A8(&lw.KProj) && isW8A8(&lw.VProj) {
			qkvOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.QProj), Scales: wmScales(&lw.QProj), Dst: q, N: lw.QProj.Rows()}
			qkvOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.KProj), Scales: wmScales(&lw.KProj), Dst: k, N: lw.KProj.Rows()}
			qkvOps[2] = linalg.W8A8Op{BQ: wmInt8(&lw.VProj), Scales: wmScales(&lw.VProj), Dst: v, N: lw.VProj.Rows()}
			matmulW8A8Batch(be, &ws, norm, K, lw.QProj.Cols(), qkvOps[:], lw.QProj.ActQuantGroup())
		} else {
			hq.prepare(be, &lw.QProj, norm, K)
			matmulPre(be, &lw.QProj, &hq, norm, q, K)
			matmulPre(be, &lw.KProj, &hq, norm, k, K)
			matmulPre(be, &lw.VProj, &hq, norm, v, K)
		}
		if arch.QKVBias {
			for i := range K {
				addBias(row(q, i, qDim), lw.QBias)
				addBias(row(k, i, kvDim), lw.KBias)
				addBias(row(v, i, kvDim), lw.VBias)
			}
		}
		invFreq := arch.ropeInvFreq(l)
		ms := arch.ropeMscale(l)
		isLocal := cache.isLocal(l)
		noPE := arch.isNoPELayer(l) // Cohere2 global layers: no positional encoding
		for i := range K {
			pos := startPos + i
			if cache.treeRowPos != nil {
				pos = cache.treeRowPos[i]
			}
			qi, ki, vi := row(q, i, qDim), row(k, i, kvDim), row(v, i, kvDim)
			if arch.QKNorm {
				if arch.QKNormWhole {
					rmsNorm(qi, lw.QNorm, 1, nH*hd, arch.NormEps, arch.RMSAddOne)
					rmsNorm(ki, lw.KNorm, 1, nKV*hd, arch.NormEps, arch.RMSAddOne)
				} else {
					rmsNorm(qi, lw.QNorm, nH, hd, arch.NormEps, arch.RMSAddOne)
					rmsNorm(ki, lw.KNorm, nKV, hd, arch.NormEps, arch.RMSAddOne)
				}
			}
			if !noPE {
				ropeAt(qi, nH, hd, pos, invFreq, ms, arch.MRopeSection, cache.mropePos, cache.mropeDelta, arch.ropeInterleave, arch.MRopeInterleaved)
				ropeAt(ki, nKV, hd, pos, invFreq, ms, arch.MRopeSection, cache.mropePos, cache.mropeDelta, arch.ropeInterleave, arch.MRopeInterleaved)
			}
			// Batched twin of causalAttention's attn-temp step (see AttnTempBeta's own comment):
			// per-position, since pos varies across this batch's K rows.
			if arch.AttnTempBeta != 0 {
				scale := float32(1 + arch.AttnTempBeta*math.Log1p(math.Floor(float64(pos)/arch.AttnTempOrigMaxPos)))
				for j := range qi {
					qi[j] *= scale
				}
			}
			if !isLocal {
				cache.Append(l, ki, vi) // global: append now; local: deferred to commitBatch below
			}
		}
		// QKᵀ and scores·V for all K positions run per head on the SIMD A·Bᵀ kernel (the L² terms), not the scalar per-position
		// attendQuery. With useAcc64 (f64 accumulation) the result is bit-identical to the sequential reference decode also runs
		// (causalAttention), so a batched verify reproduces sequential greedy exactly: required for same-model speculative decoding and
		// for the MoE top-k router never to cascade. Local layers read an assembled [base, startPos+K) window (ring history plus the K new
		// rows); the ring write is deferred until after the read so a K>W batch cannot evict in-batch history. Global layers read
		// append-forever.
		if isLocal {
			base, nRows := cache.batchReadLocal(l, startPos, K, k, v, alk, alv)
			attendBatchedHeads(q, ctx, alk[:nRows*kvDim], alv[:nRows*kvDim], base, cache, l, startPos, K, global, arch, useAcc64, attnPool)
			cache.commitBatch(l, startPos, K, k, v)
		} else if cache.quant == kvI8 {
			// int8 global: the Append loop above already quantized the new K/V into
			// the layer; dequant the full history into f32 scratch for the matmul.
			nKeys := cache.dequantGlobalLayer(l, kvDim, alk, alv)
			attendBatchedHeads(q, ctx, alk[:nKeys*kvDim], alv[:nKeys*kvDim], 0, cache, l, startPos, K, global, arch, useAcc64, attnPool)
		} else {
			attendBatchedHeads(q, ctx, cache.Keys(l), cache.Vals(l), 0, cache, l, startPos, K, global, arch, useAcc64, attnPool)
		}
		// Output gating, per row, before o_proj: the batched twin of the K=1 call in causalAttention, sharing
		// applyGateRow/applySigmoidGateRow so the two paths cannot diverge. `norm` still holds this layer's post-input-layernorm rows (it
		// is not recomputed for the MLP until after o_proj), which is the tensor the gate reads. Laguna (softplus) and Spark-X2.5
		// (sigmoid, arch.AttnGate == GateSigmoid) both come through hasAttnOutputGate.
		if arch.hasAttnOutputGate() {
			gRows := lw.GProj.Rows()
			if cap(gbuf) < K*gRows {
				gbuf = make([]float32, K*gRows)
			}
			gb := gbuf[:K*gRows]
			matmul(be, &lw.GProj, norm, gb, K)
			perHead := gRows == nH
			for i := range K {
				if arch.AttnGate == GateSigmoid {
					applySigmoidGateRow(row(gb, i, gRows), row(ctx, i, qDim), perHead, nH, hd)
				} else {
					applyGateRow(row(gb, i, gRows), row(ctx, i, qDim), perHead, nH, hd)
				}
			}
		}
		matmul(be, &lw.OProj, ctx, att, K)
		if arch.OutBias {
			for i := range K {
				addBias(row(att, i, hidden), lw.OBias)
			}
		}
		if postNorm {
			for i := range K {
				normalize(arch, row(att, i, hidden), lw.PostAttnNorm, nil, hidden)
			}
		}
		if !parallel {
			// Sequential: add the attention residual, then re-norm the updated stream for the MLP.
			addResidual(h, att)
			if !postOnly {
				for i := range K {
					normalizeInto(arch, row(norm, i, hidden), row(h, i, hidden), lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
				}
			} else {
				copy(norm, h)
			}
		}
		// Parallel (Cohere/GPT-J): `norm` still holds the single shared input norm and
		// `att` is held back — both fold into ONE residual add after the MLP below.
		if arch.MoE != nil && lw.Experts != nil {
			if parallel {
				// No parallel-block MoE family exists yet; the batched MoE branch below
				// adds only its own contribution and `continue`s, which would silently
				// drop `att`. Fail loud until a parallel+MoE family needs the joint add.
				return nil, errNotImplemented
			}
			// Sparse MoE (Mellum / Mixtral): the router selects different experts per token, so the FFN is not batchable across K and runs the
			// per-token moeMLP for each row (bit-identical to the sequential path). At agentic prompt lengths this is not where the time goes
			// (attention dominates), so batching it expert-major is not a compute lever; its case has to be made on streaming I/O, where the
			// same expert is re-fetched per row, and measured there (docs/code-notes/decoder.md#Model.runLayersFromEmbedN.moeFFN).
			//
			// GLM's dense prefix layers (Experts nil) fall through to the dense FFN below. With knobSet.moeExpertMajor on, the routed experts
			// run expert-major in chunks so each expert's weights are read once per chunk rather than once per token. moeMLPBatch refuses and
			// falls through for the order-dependent cases (test seams, a live pager, a shared expert), and is bit-identical when it runs
			// (TestMoEExpertMajor_bitIdentical).
			emDone := make([]bool, K)
			var emOut []float32
			if m.knobs.moeExpertMajor() {
				emOut = make([]float32, K*hidden)
				for c0 := 0; c0 < K; c0 += moeExpertMajorChunk {
					c1 := min(c0+moeExpertMajorChunk, K)
					ok, err := moeMLPBatch(norm[c0*hidden:c1*hidden], c1-c0, lw, arch, be, m.pager,
						emOut[c0*hidden:c1*hidden])
					if err != nil {
						return nil, err
					}
					if !ok {
						break // refused: leave every row to the per-row path below
					}
					for i := c0; i < c1; i++ {
						emDone[i] = true
					}
				}
			}
			for i := range K {
				// nil scr: this batched-prefill path builds its own per-K-batch scratch
				// above and has no cache.scr in scope; moeMLP falls back to allocating,
				// amortized over K tokens (not the flagged single-token decode hot path).
				var ff []float32
				if emDone[i] {
					ff = emOut[i*hidden : (i+1)*hidden]
				} else {
					var err error
					ff, err = moeMLP(row(norm, i, hidden), lw, arch, be, nil, m.pager)
					if err != nil {
						return nil, err
					}
				}
				if postNorm {
					normalize(arch, ff, lw.PostMLPNorm, nil, hidden)
				}
				hi := row(h, i, hidden)
				for j := range ff {
					hi[j] += ff[j]
				}
			}
			// DeepStack, as on the dense path below: after the layer's residual, before the capture. The `continue` below
			// skips the dense path's copy of both.
			if ds := cache.deepstack; ds != nil && l < len(ds.rows) {
				addDeepstack(h, ds, l, startPos, K, hidden)
			}
			// Hidden-state seam, as in the dense path below: MoE layers must also record captured[ci], or a capture against a sparse-MoE
			// target leaves captured all-nil and fuseAt slices a nil slice.
			if cache.captureLayers != nil {
				for ci, cl := range cache.captureLayers {
					if cl == l {
						cache.captured[ci] = append(cache.captured[ci][:0], h...)
					}
				}
			}
			continue
		}
		if isW8A8(&lw.GateProj) && isW8A8(&lw.UpProj) {
			guOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.GateProj), Scales: wmScales(&lw.GateProj), Dst: gate, N: lw.GateProj.Rows()}
			guOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.UpProj), Scales: wmScales(&lw.UpProj), Dst: up, N: lw.UpProj.Rows()}
			matmulW8A8Batch(be, &ws, norm, K, lw.GateProj.Cols(), guOps[:], lw.GateProj.ActQuantGroup())
		} else {
			hq.prepare(be, &lw.GateProj, norm, K)
			matmulPre(be, &lw.GateProj, &hq, norm, gate, K)
			matmulPre(be, &lw.UpProj, &hq, norm, up, K)
		}
		switch arch.Act {
		case ActGeluTanh:
			if len(gate) < activationFanoutThreshold {
				geglu(gate, up)
			} else {
				parallelElementwise(len(gate), func(lo, hi int) {
					for j := lo; j < hi; j++ {
						gate[j] = geluTanh(gate[j]) * up[j]
					}
				})
			}
		case ActSiLU:
			if len(gate) < activationFanoutThreshold {
				swiglu(gate, up)
			} else {
				parallelElementwise(len(gate), func(lo, hi int) {
					for j := lo; j < hi; j++ {
						gate[j] = silu(gate[j]) * up[j]
					}
				})
			}
		case ActGelu:
			if len(gate) < activationFanoutThreshold {
				gegluExact(gate, up)
			} else {
				parallelElementwise(len(gate), func(lo, hi int) {
					for j := lo; j < hi; j++ {
						gate[j] = geluErf(gate[j]) * up[j]
					}
				})
			}
		default:
			return nil, errNotImplemented
		}
		matmul(be, &lw.DownProj, gate, mlpOut, K)
		if postNorm {
			for i := range K {
				normalize(arch, row(mlpOut, i, hidden), lw.PostMLPNorm, nil, hidden)
			}
		}
		if parallel {
			// Single residual add: attention + MLP, both from the shared input norm.
			addResidual2(h, att, mlpOut)
		} else {
			addResidual(h, mlpOut)
		}
		if ds := cache.deepstack; ds != nil && l < len(ds.rows) {
			addDeepstack(h, ds, l, startPos, K, hidden)
		}
		// Read-only hidden-state seam (05), batched: copy all K rows of this layer's
		// output when requested. captured[ci] holds [K*hidden]. nil ⇒ zero overhead.
		if cache.captureLayers != nil {
			for ci, cl := range cache.captureLayers {
				if cl == l {
					cache.captured[ci] = append(cache.captured[ci][:0], h...)
				}
			}
		}
	}

	// Advance the cache by K explicitly: a local last layer defers its Append
	// (commitBatch doesn't touch pos), so the per-token last-layer auto-advance
	// can't be relied on. Idempotent when the last layer is global (Append already
	// stepped pos to startPos+K). canBatchN excludes the manualPos families.
	cache.advanceTo(startPos + K)

	for i := range K {
		normalize(arch, row(h, i, hidden), m.w.FinalNorm, m.w.FinalNormBias, hidden)
	}
	return h, nil
}

// attnTask is one unit of work handed to attnWorkerPool: run fn over the
// caller's own [lo,hi) slice, tagged with worker index w so fn can index its
// own per-worker scratch (pool[w] in attendGroupedLayer).
type attnTask struct {
	fn        func(w, lo, hi int)
	w, lo, hi int
	done      *sync.WaitGroup
}

// attnWorkerPool is a small set of long-lived goroutines, parked on a channel between rounds and shared by every decode step's
// attention fan-out. It replaced a fresh goroutine per split: a decoded token does up to 28 layers x 2 kv heads x 2 phases of
// fan-outs, and the spawn overhead alone outweighed most of the grouped kernel's savings. A persistent pool pays goroutine
// creation once; each round is a channel send to a parked goroutine. Record:
// docs/measurements/r13-served-decode-2026-09-20.md.
type attnWorkerPool struct {
	tasks chan attnTask
}

func newAttnWorkerPool(n int) *attnWorkerPool {
	p := &attnWorkerPool{tasks: make(chan attnTask, n)}
	for range n {
		go func() {
			for t := range p.tasks {
				t.fn(t.w, t.lo, t.hi)
				t.done.Done()
			}
		}()
	}
	return p
}

// globalAttnWorkerPool is process-wide, not per-Model: total attention fan-out parallelism is bounded by maxAttnWorkers across
// every concurrent request, the same oversubscription discipline applied within one call.
var globalAttnWorkerPool = sync.OnceValue(func() *attnWorkerPool {
	return newAttnWorkerPool(maxAttnWorkers)
})

// runSplitAligned fans work over the independent-output range [0,n) into up to `workers` contiguous, non-overlapping slices, each
// width rounded up to a multiple of align, running fn(w, lo, hi) once per non-empty slice: worker 0 inline, the rest via the
// persistent globalAttnWorkerPool. fn must write only to worker w's own scratch (pool[w]) and to its own disjoint [lo,hi) region of
// any shared output; no combining is performed, so callers must not need one.
//
// The alignment matters: aikit's NEON grouped kernels process their range in 8-wide blocks and fall back to the plain-Go grouped
// kernel (6-10x slower) for a remainder that does not fill a block. Equal-width splitting would give every worker a ragged
// remainder; rounding up keeps each slice block-aligned at the cost of slightly uneven work.
func runSplitAligned(workers, n, align int, fn func(w, lo, hi int)) {
	if workers <= 1 || n < workers {
		fn(0, 0, n)
		return
	}
	pool := globalAttnWorkerPool()
	var wg sync.WaitGroup
	per := (n + workers - 1) / workers
	if align > 1 {
		per = ((per + align - 1) / align) * align
	}
	for w := 1; w < workers; w++ {
		lo, hi := w*per, min((w+1)*per, n)
		if lo >= hi {
			continue
		}
		wg.Add(1)
		pool.tasks <- attnTask{fn: fn, w: w, lo: lo, hi: hi, done: &wg}
	}
	hi0 := min(per, n)
	fn(0, 0, hi0)
	wg.Wait()
}

// attendTileFor is attendTileForK reading the live environment (tests that build scratch by hand).
func attendTileFor(ws *headWorkerScratch, K, nKeys, hd int) int {
	return attendTileForK(nil, ws, K, nKeys, hd)
}

// attendTileForK returns how many query rows attendOneHead may process at once with this slot: the cache-sized tile from
// attnRowTileK, clamped to what the slot's buffers hold. The tile is a property of the pool, not of the call: the pool is sized
// once from maxKeys = startPos+K, but a local layer whose ring has wrapped assembles a shorter window and attnRowTile is inverse
// in nKeys, so the per-layer tile can exceed the qh the slot was allocated and the Q gather would slice past its length. The
// slot's capacity is the binding constraint, so the slot decides: with kt <= tile and nKeys <= maxKeys every other slot buffer
// (scores, ch, kh, vt) fits once qh does. It also covers hand-built scratch in the ring tests.
func attendTileForK(k *knobSet, ws *headWorkerScratch, K, nKeys, hd int) int {
	tile := attnRowTileK(k, K, nKeys)
	if hd < 1 {
		return max(1, tile)
	}
	if rows := len(ws.qh) / hd; rows >= 1 && tile > rows {
		tile = rows
	}
	return max(1, tile)
}

// attendBatchedHeads computes grouped-query causal attention for K query positions at once, per head: per KV head it gathers
// K_head and V_headᵀ once (f32 path), then per query head scores = Q·Kᵀ, a scaled causal/window-masked softmax per row (masked
// entries zeroed so they drop out of the next matmul), and ctx = scores·V scattered into ctx[K,qDim].
//
// With useAcc64 the dots accumulate in f64 and are bit-identical to the sequential attendQuery/causalAttention. The f32 path
// (MatmulBT) reassociates the reduction, so its parity is argmax-exact plus cosine, the standard the GPU residency attention also
// meets. The softmax exp stays per row in float64. Scratch slices (qh, kh, vt, scores, ch) are caller-owned and reused across
// layers.
//
// keys/vals are the contiguous K/V the gather reads, physical row 0 holding absolute key position `base`: cache.Keys(layer) at
// base 0 for a global layer; for a local (sliding-window) layer an assembled [base, startPos+K) window, so the ring's wrap is
// invisible here and the math is byte-identical to append-forever. Per-query masking stays in absolute positions
// (WindowStart/attendHi), mapped to physical columns by s-base.
func attendBatchedHeads(q, ctx, keys, vals []float32, base int, cache *KVCache, layer, startPos, K int, global bool, arch *Architecture, useAcc64 bool, pool []headWorkerScratch) {
	// headsAt, not NumHeads: Laguna varies the query head count per layer (its KV heads stay uniform, so group is per-layer too).
	nH, nKV, hd := arch.headsAt(layer), arch.NumKVHeads, arch.HeadDim
	kvDim, qDim := nKV*hd, nH*hd
	group := nH / nKV
	scale := arch.AttnScale
	nKeys := len(keys) / kvDim
	// MoE routing is discontinuous: an f32 QKᵀ reassociation can flip a top-k expert at a near-tie and cascade, so the acc64 kernels
	// accumulate each dot in f64 (bit-identical to the sequential f64 reference); dense MLPs tolerate the f32 error. There is no
	// shared `matmul` variable: acc64 calls MatmulQKAcc64/MatmulAVAcc64 directly (strided, no gather), and the f32 path is handed its
	// matmul per head (package-level MatmulBT when the head loop is serial, the worker's serial Workspace when not).

	// The fused schedule is eligible only on the f32 path (it would break acc64's bit-identity), without a tree mask or logit softcap.
	// Each worker uses its own ws.fused, never a shared one, since each gathers a different kv head's V.
	fusedOK := !useAcc64 && cache.treeMask == nil && arch.AttnLogitSoftcap == 0
	// attendOneHead runs one query head's QKᵀ → softmax → scores·V → scatter into ctx using ws's scratch. It runs concurrently
	// across heads and is bit-identical regardless of pool slot, goroutine or finish order: each head's reduction order is
	// unchanged and its ctx write is a disjoint qhead*hd slice. mm is the f32 matmul to use (package-level MatmulBT on the serial
	// arm, the worker's own serial Workspace on the head-parallel arm); it is unused on the acc64 path.
	attendOneHead := func(qhead int, ws *headWorkerScratch, mm func(a, b, dst []float32, M, K, N int)) {
		var fs *fusedScratch
		if fusedOK {
			fs = ws.fused
		}
		kvh := qhead / group
		qh, scores, ch, avAcc := ws.qh, ws.scores, ws.ch, ws.avAcc
		// Walk the query rows in tiles. Every step below is row-wise, so splitting rows splits independent outputs, which is what
		// bit-identity permits. No key-dimension split happens here and none may: it would re-associate the softmax denominator and the AV
		// fold, exactly what acc64 exists to prevent. The point is memory: scores is tile*nKeys, not K*nKeys, so a worker slot stops
		// growing with the square of the prompt and the pool can still fan out on a long prompt.
		//
		// `i` indexes the tile; `gi` is the global row. Positions and masks use `gi` (startPos+gi, treeRowPos[gi], treeMask[gi]), buffers
		// use `i`. The tile comes from attendTileForK, not attnRowTile: the slot's capacity binds, and recomputing it from this layer's key
		// count panicked a warm windowed session.
		tile := attendTileForK(arch.knobs, ws, K, nKeys, hd)
		for t0 := 0; t0 < K; t0 += tile {
			kt := min(tile, K-t0)
			for i := range kt { // gather this tile's Q_head [kt,hd]
				b := (t0+i)*qDim + qhead*hd
				copy(qh[i*hd:i*hd+hd], q[b:b+hd])
			}
			// The fused schedule, when enabled and applicable, replaces the whole QKᵀ/softmax/scores·V sequence for this tile and keeps the
			// score block resident instead of materializing kt x nKeys (fusedattn.go). It declines, and falls through, for acc64 (whose
			// bit-identity it would break) and for tree attention (whose per-(row,column) mask is not the contiguous [lo,hi] bound it handles).
			if fs != nil {
				for i := range kt {
					gi := t0 + i
					pos := startPos + gi
					fs.lo[i] = cache.WindowStart(pos, global) - base
					fs.hi[i] = cache.attendHi(pos) - base
				}
				if attendTileFused(mm, qh[:kt*hd], ws.kh[:nKeys*hd], fs.vBlk, ch[:kt*hd],
					fs.sBlk, fs.tmp, fs.acc, fs.mRun, fs.lRun,
					kt, hd, nKeys, scale, fs.lo[:kt], fs.hi[:kt]) {
					for i := range kt { // scatter this tile's ctx_head into ctx[K,qDim]
						b := (t0+i)*qDim + qhead*hd
						copy(ctx[b:b+hd], ch[i*hd:i*hd+hd])
					}
					continue
				}
			}
			// QKᵀ: scores[kt,nKeys] = Q_head[kt,hd] · K_head[nKeys,hd]ᵀ. Acc64 reads keys directly (row stride kvDim, element stride 1),
			// skipping the kh gather. MatmulQKAcc64 interleaves 8 keys' dot products as concurrent f64 accumulator chains to hide FMA latency;
			// each key's own d-order fold is unchanged, so the result is bit-identical to the plain strided reduction
			// (TestAttendStrided_matchesGatherReference checks it at this stride).
			if useAcc64 {
				linalg.MatmulQKAcc64(qh[:kt*hd], keys, scores[:kt*nKeys], kt, hd, nKeys, kvh*hd, kvDim)
			} else {
				mm(qh[:kt*hd], ws.kh[:nKeys*hd], scores[:kt*nKeys], kt, hd, nKeys)
			}
			// Scaled, masked softmax per query row; zero the out-of-range entries
			// so they contribute nothing to the scores·V matmul below.
			for i := range kt {
				gi := t0 + i // global row: positions and masks are indexed by it, buffers by i
				pos := startPos + gi
				if cache.treeRowPos != nil {
					pos = cache.treeRowPos[gi]
				}
				rowS := scores[i*nKeys : i*nKeys+nKeys]
				// TREE attention (05): row i attends to the whole committed prefix
				// [loP, batchCol0) plus only its ancestor batch columns (treeMask[i][j]).
				if cache.treeMask != nil {
					loP := cache.WindowStart(pos, global) - base
					batchCol0 := startPos - base
					allowed := func(s int) bool {
						if s < batchCol0 {
							return s >= loP
						}
						j := s - batchCol0
						return j < K && cache.treeMask[gi][j]
					}
					maxS := math.Inf(-1)
					for s := range nKeys {
						if allowed(s) {
							sc := softcapScore(float64(rowS[s])*scale, arch.AttnLogitSoftcap)
							rowS[s] = float32(sc)
							if sc > maxS {
								maxS = sc
							}
						}
					}
					var sum float64
					for s := range nKeys {
						if allowed(s) {
							e := math.Exp(float64(rowS[s]) - maxS)
							rowS[s] = float32(e)
							sum += e
						} else {
							rowS[s] = 0
						}
					}
					inv := 1.0 / sum
					for s := range nKeys {
						if rowS[s] != 0 {
							rowS[s] = float32(float64(rowS[s]) * inv)
						}
					}
					continue
				}
				// Absolute attend range [start, hi]; map to physical columns by −base
				// (base = absolute position of column 0). hi is the inclusive upper
				// key bound: pos for a causal text query, or the image-block end for a
				// bidirectional image position (so it also attends to the block's
				// future tokens). Equals pos with no image blocks — inert for text.
				loP := cache.WindowStart(pos, global) - base
				hiP := cache.attendHi(pos) - base
				if loP < 0 {
					loP = 0
				}
				if hiP >= nKeys {
					hiP = nKeys - 1
				}
				if loP <= hiP {
					active := rowS[loP : hiP+1]
					_ = active[len(active)-1]
					maxS := math.Inf(-1)
					for s, v := range active {
						sc := softcapScore(float64(v)*scale, arch.AttnLogitSoftcap)
						active[s] = float32(sc)
						if sc > maxS {
							maxS = sc
						}
					}
					var sum float64
					for s, v := range active {
						e := math.Exp(float64(v) - maxS)
						active[s] = float32(e)
						sum += e
					}
					inv := 1.0 / sum
					clear(rowS[:loP])
					for s, v := range active {
						active[s] = float32(float64(v) * inv)
					}
					clear(rowS[hiP+1 : nKeys])
				} else {
					clear(rowS[:nKeys])
				}
			}
			// scores·V: ctx_head[kt,hd] = scores[kt,nKeys] · V_head[nKeys,hd]. Acc64 reads vals directly, "as if transposed" (row stride 1,
			// element stride kvDim), skipping a vt gather and transpose. MatmulAVAcc64 reads V rows contiguously (keys-outer, dims-inner) into
			// hd independent f64 accumulators; each dim's accumulator sees the same key-ascending adds as the strided walk, so it is
			// bit-identical.
			//
			// A one-row tile (every K=1 decode) writes this head's context straight into ctx: MatmulAVAcc64 overwrites its destination and the
			// head's hd floats at ctx[t0*qDim+qhead*hd] are contiguous, so the scratch `ch` and the scatter copy are dead work. A multi-row
			// tile's [kt,hd] result is not contiguous in ctx (row stride qDim) and keeps the scatter.
			direct := useAcc64 && kt == 1
			if direct {
				b := t0*qDim + qhead*hd
				linalg.MatmulAVAcc64(scores[:nKeys], vals, ctx[b:b+hd], avAcc, 1, nKeys, hd, kvh*hd, kvDim)
			} else if useAcc64 {
				linalg.MatmulAVAcc64(scores[:kt*nKeys], vals, ch[:kt*hd], avAcc, kt, nKeys, hd, kvh*hd, kvDim)
			} else {
				mm(scores[:kt*nKeys], ws.vt[:hd*nKeys], ch[:kt*hd], kt, nKeys, hd)
			}
			if !direct {
				for i := range kt { // scatter this tile's ctx_head into ctx[K,qDim]
					b := (t0+i)*qDim + qhead*hd
					copy(ctx[b:b+hd], ch[i*hd:i*hd+hd])
				}
			}
		}
	}

	// attendGroupedHeads is the grouped-kernel path: one MatmulQKAcc64Group/MatmulAVAcc64Group pair covers all attnGroupedNEONSize
	// query heads of one kv head, instead of that many attendOneHead calls re-reading the same K/V rows. The caller must have checked
	// K==1 (decode; M>1 is not wired), no tree mask (its per-(row,column) mask needs attendOneHead's own branch), and
	// group == attnGroupedNEONSize.
	//
	// Q for these heads is contiguous in q's [K,qDim] layout (and so is ctx), so no gather is needed. Softmax stays per head, row by
	// row, identical to attendOneHead's non-tree branch: grouping shares the QKᵀ and scores·V loads and folds, never the reduction
	// inside each head's softmax.
	attendGroupedHeads := func(kvh int, ws *headWorkerScratch) {
		qh0 := kvh * group
		a := q[qh0*hd : qh0*hd+group*hd]
		gScores := ws.groupScores[:group*nKeys]
		linalg.MatmulQKAcc64Group(a, keys, gScores, group, hd, nKeys, kvh*hd, kvDim)
		pos := startPos // K==1 is a precondition, so this call's only row is gi=0
		loP := cache.WindowStart(pos, global) - base
		hiP := cache.attendHi(pos) - base
		if loP < 0 {
			loP = 0
		}
		if hiP >= nKeys {
			hiP = nKeys - 1
		}
		for g := range group {
			rowS := gScores[g*nKeys : g*nKeys+nKeys]
			if loP <= hiP {
				active := rowS[loP : hiP+1]
				_ = active[len(active)-1]
				maxS := math.Inf(-1)
				for s, v := range active {
					sc := softcapScore(float64(v)*scale, arch.AttnLogitSoftcap)
					active[s] = float32(sc)
					if sc > maxS {
						maxS = sc
					}
				}
				var sum float64
				for s, v := range active {
					e := math.Exp(float64(v) - maxS)
					active[s] = float32(e)
					sum += e
				}
				inv := 1.0 / sum
				clear(rowS[:loP])
				for s, v := range active {
					active[s] = float32(float64(v) * inv)
				}
				clear(rowS[hiP+1 : nKeys])
			} else {
				clear(rowS[:nKeys])
			}
		}
		// MatmulAVAcc64Group overwrites a [group, hd] destination and this kv group's query heads are contiguous in ctx, so it writes
		// straight into ctx with no scratch copy.
		gAvAcc := ws.groupAvAcc[:group*hd]
		linalg.MatmulAVAcc64Group(gScores, vals, ctx[qh0*hd:qh0*hd+group*hd], gAvAcc, group, nKeys, hd, kvh*hd, kvDim)
		atomic.AddInt64(&attnGroupedRuns, 1)
	}
	// attnGroupedOK is this call's eligibility for the grouped path, checked once per call since none of it depends on qhead.
	// treeMask excludes speculative verify (attendGroupedHeads has no per-(row,column) mask); K != 1 excludes prefill/batched M>1
	// (not wired); attnGroupedKernels is the per-architecture platform gate.
	attnGroupedOK := useAcc64 && K == 1 && cache.treeMask == nil && attnGroupedKernels &&
		group == attnGroupedNEONSize && nKeys >= attnGroupedMinKeys && arch.knobs.attnGrouped()
	// runHeadRange walks qhead across [h0,h1), taking the grouped path for any run of attnGroupedNEONSize heads that starts on a
	// kv-group boundary and fits inside [h0,h1); a worker whose range splits a kv group uses attendOneHead for that group's heads.
	runHeadRange := func(ws *headWorkerScratch, h0, h1 int) {
		qhead := h0
		for qhead < h1 {
			if attnGroupedOK && qhead%group == 0 && qhead+group <= h1 {
				attendGroupedHeads(qhead/group, ws)
				qhead += group
				continue
			}
			attendOneHead(qhead, ws, nil)
			qhead++
		}
	}

	// attendGroupedLayer is the grouped path for a pool of more than one slot (decode, K==1). Arm A (runHeadRange/attendGroupedHeads)
	// needs one worker to own a whole kv group's heads and so collapses worker count to nKV; this keeps the pool busy and still uses
	// the grouped kernel by splitting within each kv head's group: QK by key range, softmax by row, AV by dim range, the three phases
	// joined in between.
	//
	// All three are independent-output splits, not reduction splits (QK reduces over d and is split by key; AV reduces over keys and
	// is split by dim), so no floating-point combining is needed: each worker's slice lands in a disjoint region of the shared buffer
	// by plain copy. The result is bit-identical to the per-head path and to Arm A; TestAttendGroupedLayer_manyWorkers is the check.
	// The softmax phase must stay split: see the note on it below.
	attendGroupedLayer := func(pool []headWorkerScratch) {
		splitWorkers := max(min(len(pool), maxAttnWorkers), 1)
		leader := &pool[0]
		if c := nKV * group * nKeys; cap(leader.groupScoresCombined) < c { // grows by nKV*group a token: headroom (R-17)
			leader.groupScoresCombined = make([]float32, growCap(cap(leader.groupScoresCombined), c))
		}
		fullScores := leader.groupScoresCombined[:nKV*group*nKeys]
		fullCtx := ctx[:nKV*group*hd] // R-17: kv head kvh's group is ctx[kvh*group*hd:(kvh+1)*group*hd], the layout the old combined scratch was copied into at the end

		runSplitAligned(splitWorkers, nKeys, attnGroupedNEONBlock, func(w, lo, hi int) {
			ws := &pool[w]
			sliceLen := hi - lo
			slice := ws.groupScores[:group*sliceLen]
			for kvh := range nKV {
				qh0 := kvh * group
				a := q[qh0*hd : qh0*hd+group*hd]
				linalg.MatmulQKAcc64Group(a, keys, slice, group, hd, sliceLen, kvh*hd+lo*kvDim, kvDim)
				dst := fullScores[kvh*group*nKeys : (kvh+1)*group*nKeys]
				for g := range group {
					copy(dst[g*nKeys+lo:g*nKeys+hi], slice[g*sliceLen:(g+1)*sliceLen])
				}
			}
		})

		pos := startPos
		loP := cache.WindowStart(pos, global) - base
		hiP := cache.attendHi(pos) - base
		if loP < 0 {
			loP = 0
		}
		if hiP >= nKeys {
			hiP = nKeys - 1
		}
		// Softmax is split by row (one kv head's one query head) across the workers, as the per-head path already does inside each head's
		// worker. Running it serially on this goroutine was measured slower than not grouping at all; do not collapse it
		// (docs/code-notes/decoder.md#attendGroupedLayer.softmax).
		runSplitAligned(splitWorkers, nKV*group, 1, func(w, lo, hi int) {
			for row := lo; row < hi; row++ {
				kvh, g := row/group, row%group
				rowS := fullScores[kvh*group*nKeys+g*nKeys : kvh*group*nKeys+g*nKeys+nKeys]
				if loP <= hiP {
					active := rowS[loP : hiP+1]
					_ = active[len(active)-1]
					maxS := math.Inf(-1)
					for s, v := range active {
						sc := softcapScore(float64(v)*scale, arch.AttnLogitSoftcap)
						active[s] = float32(sc)
						if sc > maxS {
							maxS = sc
						}
					}
					var sum float64
					for s, v := range active {
						e := math.Exp(float64(v) - maxS)
						active[s] = float32(e)
						sum += e
					}
					inv := 1.0 / sum
					clear(rowS[:loP])
					for s, v := range active {
						active[s] = float32(float64(v) * inv)
					}
					clear(rowS[hiP+1 : nKeys])
				} else {
					clear(rowS[:nKeys])
				}
			}
		})

		runSplitAligned(splitWorkers, hd, attnGroupedNEONBlock, func(w, d0, d1 int) {
			ws := &pool[w]
			sliceHd := d1 - d0
			sliceCtx := ws.groupCtx[:group*sliceHd]
			sliceAcc := ws.groupAvAcc[:group*sliceHd]
			for kvh := range nKV {
				rows := fullScores[kvh*group*nKeys : (kvh+1)*group*nKeys]
				linalg.MatmulAVAcc64Group(rows, vals, sliceCtx, sliceAcc, group, nKeys, sliceHd, kvh*hd+d0, kvDim)
				dst := fullCtx[kvh*group*hd : (kvh+1)*group*hd]
				for g := range group {
					copy(dst[g*hd+d0:g*hd+d1], sliceCtx[g*sliceHd:(g+1)*sliceHd])
				}
			}
		})

		atomic.AddInt64(&attnGroupedRuns, int64(nKV))
	}

	if !useAcc64 {
		// f32 path (the default for prefill above fastAttnMinPrompt). gatherKV fills ws's kh/vt for one kv head. It is a pure function of
		// (keys, vals, kvh), so two workers gathering the same kvh into their own buffers produce identical bytes, which is why the serial
		// and fan-out arms below are bit-identical, not merely close (TestAttendF32Fanout_bitIdentical).
		gatherKV := func(ws *headWorkerScratch, kvh int) {
			// The fused schedule needs V block-major (a key-range slice of the [hd, nKeys]
			// layout is not contiguous), so the layout is chosen here at gather time rather
			// than re-transposed per block.
			if fusedOK && ws.fused != nil {
				gatherKVFused(ws.kh, ws.fused.vBlk, keys, vals, kvh, hd, kvDim, nKeys)
				return
			}
			for s := range nKeys {
				kvBase := s*kvDim + kvh*hd
				copy(ws.kh[s*hd:s*hd+hd], keys[kvBase:kvBase+hd])
				vrow := vals[kvBase : kvBase+hd]
				_ = vrow[hd-1]
				_ = ws.vt[(hd-1)*nKeys+s]
				for d := range hd {
					ws.vt[d*nKeys+s] = vrow[d]
				}
			}
		}
		// Fan out over query heads, as the acc64 path does. Every pool slot already owns a full-size kh/vt pair (prefillAttnWorkersK
		// budgets 2*nKeys*hd per slot), so a worker gathers into its own buffers and nothing is shared across the group. Heads are
		// assigned in contiguous runs so a worker walks whole kv groups and re-gathers only when kvh changes: at most nKV + workers
		// gathers, not nH. MatmulBT already fans out over output columns internally; head-level fan-out is what parallelises the gather,
		// the softmax and the scatter.
		workers := 1
		if len(pool) > 1 && nH > 1 && K*nKeys >= attnHeadsParThreshold {
			workers = min(len(pool), nH)
		}
		if workers <= 1 {
			// Serial arm: one slot, and the matmul keeps its OWN column-level
			// fan-out (package-level MatmulBT) since no head-level fan-out is
			// competing with it. This is the pre-A3 behaviour exactly.
			ws := &pool[0]
			for kvh := range nKV {
				gatherKV(ws, kvh)
				for g := range group {
					attendOneHead(kvh*group+g, ws, linalg.MatmulBT)
				}
			}
			return
		}
		var wg sync.WaitGroup
		headsPer := (nH + workers - 1) / workers
		for w := 1; w < workers; w++ {
			h0, h1 := w*headsPer, min((w+1)*headsPer, nH)
			if h0 >= h1 {
				continue
			}
			wg.Add(1)
			go func(w, h0, h1 int) {
				defer wg.Done()
				ws := &pool[w]
				lastKVH := -1
				for qhead := h0; qhead < h1; qhead++ {
					if kvh := qhead / group; kvh != lastKVH {
						gatherKV(ws, kvh)
						lastKVH = kvh
					}
					// Serial matmul: the fan-out is at the head level here, and
					// nesting MatmulBT's column fan-out inside it would
					// oversubscribe. Bit-identical to the column-parallel form
					// by MatmulBT's width contract.
					attendOneHead(qhead, ws, ws.mmWS.MatmulBT)
				}
			}(w, h0, h1)
		}
		h1_0 := min(headsPer, nH)
		ws0 := &pool[0]
		lastKVH0 := -1
		for qhead := range h1_0 {
			if kvh := qhead / group; kvh != lastKVH0 {
				gatherKV(ws0, kvh)
				lastKVH0 = kvh
			}
			attendOneHead(qhead, ws0, ws0.mmWS.MatmulBT)
		}
		wg.Wait()
		return
	}

	// With more than one pool slot, Arm B (attendGroupedLayer) takes priority over Arm A, whose one-worker-per-group rule would strand
	// the other slots idle. With one slot Arm B has nothing to parallelise and the serial arm below already takes the grouped kernel
	// (runHeadRange's [0,nH) range spans whole groups), so Arm B is skipped rather than adding fork-join cost.
	if attnGroupedOK && len(pool) > 1 {
		attendGroupedLayer(pool)
		return
	}

	// The acc64 path (the one every live caller takes): the nH query heads are independent (disjoint ctx writes, no shared mutable
	// state; kh/vt are unused here). Below the fan-out floor, or with one pool slot or one head, run serially through pool[0]: a
	// fork-join costs real time the per-head kernels have already shrunk to a few microseconds a head at shallow depth.
	if len(pool) <= 1 || nH <= 1 || K*nKeys < attnHeadsParThreshold {
		runHeadRange(&pool[0], 0, nH)
		return
	}
	workers := min(len(pool), nH)
	var wg sync.WaitGroup
	headsPer := (nH + workers - 1) / workers
	for w := 1; w < workers; w++ {
		h0, h1 := w*headsPer, min((w+1)*headsPer, nH)
		if h0 >= h1 {
			continue
		}
		wg.Add(1)
		go func(w, h0, h1 int) {
			defer wg.Done()
			runHeadRange(&pool[w], h0, h1)
		}(w, h0, h1)
	}
	h1_0 := min(headsPer, nH)
	runHeadRange(&pool[0], 0, h1_0)
	wg.Wait()
}

// lmHeadN projects M post-final-norm hidden rows (h is [M, HiddenDim]) to logits
// [M, VocabSize] (+ final-logit softcap), at M=K so the head weights stream once.
func (m *Model) lmHeadN(h []float32, M int) []float32 {
	arch := m.w.arch
	logits := make([]float32, M*arch.VocabSize)
	if arch.TiedLMHead {
		matmul(m.be, &m.w.Embed, h, logits, M)
	} else {
		matmul(m.be, &m.w.LMHead, h, logits, M)
	}
	if arch.FinalLogitSoftcap > 0 {
		sc := float32(arch.FinalLogitSoftcap)
		for j, val := range logits {
			logits[j] = sc * float32(math.Tanh(float64(val/sc)))
		}
	}
	// logit_scale (Cohere multiplier stored as goinfer's reciprocal; Granite logits_scaling divisor): mirrors logitsFromHidden's tail.
	// The sequential path applies it, so batched prefill/verify must too or forwardN diverges for any LogitScale family.
	if arch.LogitScale != 0 && arch.LogitScale != 1 {
		inv := float32(1 / arch.LogitScale)
		for j := range logits {
			logits[j] *= inv
		}
	}
	return logits
}

// forwardN runs a batched forward over ids and returns the logits at every
// position ([K][VocabSize]) — used by the speculative verifier. Bit-identical to
// K sequential forwards. Falls back to sequential for the non-batched archs.
func (m *Model) forwardN(reqCtx context.Context, ids []int, cache *KVCache) ([][]float32, error) {
	// Speculative VERIFY: never fast, whatever the operator asked for. See forwardNAttn.
	return m.forwardNAttn(reqCtx, ids, cache, false)
}

// forwardNAttn is forwardN with the attention-kernel choice made by the caller. Speculative verify must run the exact kernel
// (fastAttn=false) on both arms or its equality argument collapses. A caller that must produce KV identical to Generate's prefill
// passes cpuFastAttention(), as prefillLogits does; otherwise "token-identical to plain greedy" stops holding at temperature 0 for
// any prompt over the fastAttnMinPrompt floor.
func (m *Model) forwardNAttn(reqCtx context.Context, ids []int, cache *KVCache, fastAttn bool) ([][]float32, error) {
	K := len(ids)
	if K == 0 {
		return nil, nil
	}
	// Tree verify needs the batched path: the sequential fallback ignores treeRowPos/
	// treeMask, so tree nodes would be attended as a linear chain (wrong parents) —
	// error rather than silently mis-verify.
	if cache.treeMask != nil && !m.canBatchN(K) {
		return nil, fmt.Errorf("decoder.forwardN: tree verify unsupported on this arch (not batchable)")
	}
	// Compute-time LoRA is wired only into the sequential forward, so an active adapter must take the M=1 path (as prefillLogits
	// does): the batched verify would project every position with the base model and commit base K/V, silently verifying drafts
	// against the wrong model.
	if cache.lora != nil || !m.canBatchN(K) {
		out := make([][]float32, K)
		for i, id := range ids {
			l, err := m.forward(id, cache)
			if err != nil {
				return nil, err
			}
			out[i] = append([]float32(nil), l...) // forward reuses scr.logits — copy
		}
		return out, nil
	}
	h, err := m.forwardLayersN(reqCtx, ids, cache, fastAttn)
	if err != nil {
		return nil, err
	}
	vocab := m.w.arch.VocabSize
	logits := m.lmHeadN(h, K)
	out := make([][]float32, K)
	for i := range K {
		out[i] = logits[i*vocab : i*vocab+vocab]
	}
	return out, nil
}

// prefillLogits processes the whole prompt and returns the logits at its last position (the seed for the first generated token).
// On the batched archs it runs the layers at M=len(prompt) in one pass (each weight streamed once, reused across all positions)
// and the LM head on the last position only. Otherwise it falls back to sequential runLayers + forward. Bit-identical to the
// sequential prefill when the exact kernel is in use; the cache is filled with the whole prompt either way.
func (m *Model) prefillLogits(ctx context.Context, prompt []int, cache *KVCache) ([]float32, error) {
	// Compute-time LoRA (#7) is wired only into the sequential forward (causalAttention
	// + gatedMLP), so an active adapter takes the M=1 path — the prompt's K/V must carry
	// the delta or decode would continue a base-projected context. The RAM-density win
	// (N adapters share one base) is unaffected; only adapter'd prefill speed regresses.
	if cache.lora != nil || !m.canBatchN(len(prompt)) {
		for _, id := range prompt[:len(prompt)-1] {
			// The sequential fallback checks the context per token: it has no layer batch to bound, and a LoRA'd or non-batchable arch
			// prefills here.
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if _, err := m.runLayers(id, cache); err != nil {
				return nil, err
			}
		}
		return m.forward(prompt[len(prompt)-1], cache)
	}
	h, err := m.forwardLayersN(ctx, prompt, cache, m.cpuFastAttention())
	if err != nil {
		return nil, err
	}
	hidden := m.w.arch.HiddenDim
	last := h[(len(prompt)-1)*hidden:] // [HiddenDim] — LM head on the last row only
	return m.lmHeadN(last, 1), nil
}

// prefillLogitsVL prefills a multimodal prompt and returns the last-position
// logits: the text token embeddings with the projected vision embeddings
// (imageEmbeds, [imgLen*HiddenDim]) substituted at the <image> placeholder run
// [imgPos, imgPos+imgLen), under a bidirectional attention mask over that block
// (so the image tokens see each other). Requires the batched path — the
// bidirectional image-block attention lives in attendBatchedHeads. The injected
// embeddings are RAW (the projector output), matching HF's masked_scatter, which
// overwrites the scaled placeholder embed. See docs/multimodal.md §4–5.
func (m *Model) prefillLogitsVL(ctx context.Context, ids []int, imageEmbeds []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsVLSpans(ctx, ids, []ImageSpan{{Pos: imgPos, Len: imgLen}}, imageEmbeds, cache, false)
}

// prefillLogitsVLSpans is prefillLogitsVL for several images (S11): each span is its own bidirectional block, and
// imageEmbeds holds every span's rows concatenated in span order.
// causal (Pixtral, S10) splices the spans without a bidirectional block: plain causal attention over the prompt.
func (m *Model) prefillLogitsVLSpans(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache, causal bool) ([]float32, error) {
	hN, err := m.prefillHiddenVLSpans(ctx, ids, spans, imageEmbeds, cache, causal)
	if err != nil {
		return nil, err
	}
	hidden := m.w.arch.HiddenDim
	return m.lmHeadN(hN[(len(ids)-1)*hidden:], 1), nil
}

// prefillHiddenVLSpans runs prefillLogitsVLSpans's prefill and returns the final hidden rows of every position (before
// the LM head), so a test can grade every position, not only the last, through the production path.
func (m *Model) prefillHiddenVLSpans(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache, causal bool) ([]float32, error) {
	if causal && m.w.arch.lfm2 != nil { // S10, LFM2-VL: lfm2 has no batched pass; its own forward, one row at a time
		return m.prefillHiddenLFM2VL(ctx, ids, spans, imageEmbeds, cache)
	}
	if !m.canBatchN(len(ids)) {
		return nil, fmt.Errorf("decoder: multimodal prefill needs the batched path (canBatchN false)")
	}
	hidden := m.w.arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageEmbeds, hidden); err != nil {
		return nil, err
	}
	h := m.embedN(ids)
	spliceImageSpans(h, spans, imageEmbeds, hidden) // raw projected features, no embed scale
	if !causal {
		cache.SetImageBlocks(imageSpanBlocks(spans))
	}
	return m.runLayersFromEmbedN(ctx, h, cache, m.cpuFastAttention())
}

// prefillHiddenLFM2VL is prefillHiddenVLSpans for lfm2 (LFM2-VL, S10): the text embeddings with the image rows spliced
// in, each run through runLayersLFM2FromEmbed in order (the conv windows see every row, image rows included), and each
// position's hidden state final-normed, as the batched entry returns them.
func (m *Model) prefillHiddenLFM2VL(ctx context.Context, ids []int, spans []ImageSpan, imageEmbeds []float32, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	hidden := arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageEmbeds, hidden); err != nil {
		return nil, err
	}
	h := m.embedN(ids)
	spliceImageSpans(h, spans, imageEmbeds, hidden)
	for i := range ids {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		row := h[i*hidden : (i+1)*hidden]
		out, err := m.runLayersLFM2FromEmbed(row, cache)
		if err != nil {
			return nil, err
		}
		normalize(arch, out, m.w.FinalNorm, m.w.FinalNormBias, hidden)
	}
	return h, nil
}

// prefillLogitsQwenVL prefills a Qwen2.5-VL multimodal prompt and returns the
// last-position logits: text embeddings with the merged vision features
// (imageFeats, [imgLen*HiddenDim] from the ViT+merger) substituted at the <image>
// run [imgPos,imgPos+imgLen), under m-RoPE 3D positions (mropePos, one (t,h,w) per
// absolute sequence position). Two deltas vs prefillLogitsVL (Gemma 3): the image
// tokens attend CAUSALLY — no bidirectional image block (Qwen's bidirectionality is
// inside the ViT, not the decoder) — and the rotary uses m-RoPE (ropeAt reads
// cache.mropePos). The merged features are RAW (no embed scale), matching HF's
// scatter into inputs_embeds.
func (m *Model) prefillLogitsQwenVL(ctx context.Context, ids []int, imageFeats []float32, imgPos, imgLen int, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	return m.prefillLogitsQwenVLSpans(ctx, ids, []ImageSpan{{Pos: imgPos, Len: imgLen}}, imageFeats, mropePos, cache)
}

// prefillLogitsQwenVLSpans is prefillLogitsQwenVL for several images (S11): every span's merged rows, concatenated in
// span order in imageFeats, replace that span's placeholder run, and mropePos covers them all (mropePositions with one
// grid per image).
func (m *Model) prefillLogitsQwenVLSpans(ctx context.Context, ids []int, spans []ImageSpan, imageFeats []float32, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	if m.w.arch.qwen35 != nil { // the Gated-DeltaNet hybrids take their own prefill (batched when qwen35BatchNAnyPos applies)
		return m.prefillLogitsQwen35VL(ctx, ids, imageFeats, spans, mropePos, cache)
	}
	if !m.canBatchN(len(ids)) {
		return nil, fmt.Errorf("decoder: Qwen2.5-VL prefill needs the batched path (canBatchN false)")
	}
	hidden := m.w.arch.HiddenDim
	if err := checkImageSpans(spans, len(ids), imageFeats, hidden); err != nil {
		return nil, err
	}
	if len(mropePos) != len(ids) {
		return nil, fmt.Errorf("decoder: mropePos len %d, want %d (one per token)", len(mropePos), len(ids))
	}
	h := m.embedN(ids)
	spliceImageSpans(h, spans, imageFeats, hidden)    // raw merged features, no embed scale
	cache.mropePos = mropePos                         // ropeAt switches to m-RoPE for this prefill
	cache.mropeDelta = mropeDelta(mropePos, len(ids)) // decode past the prefill rotates at seqPos+delta
	hN, err := m.runLayersFromEmbedN(ctx, h, cache, m.cpuFastAttention())
	if err != nil {
		return nil, err
	}
	return m.lmHeadN(hN[(len(ids)-1)*hidden:], 1), nil
}

// addDeepstack adds DeepStack set l to the rows of h ([K*hidden], positions startPos..startPos+K) that fall in the image
// run (S10): HF's hidden_states[visual_pos_masks] += deepstack_visual_embeds[l], after decoder layer l.
func addDeepstack(h []float32, ds *deepstackRows, l, startPos, K, hidden int) {
	set := ds.rows[l]
	off := 0 // the first row of this span within the set
	for _, s := range ds.spans {
		for p := max(s.Pos, startPos); p < min(s.Pos+s.Len, startPos+K); p++ {
			dst := h[(p-startPos)*hidden : (p-startPos+1)*hidden]
			src := set[(off+p-s.Pos)*hidden : (off+p-s.Pos+1)*hidden]
			for j := range dst {
				dst[j] += src[j]
			}
		}
		off += s.Len
	}
}

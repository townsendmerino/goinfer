package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"sync"

	"github.com/townsendmerino/aikit/linalg"
)

// cpuFastAttention reports whether the operator opted into A3's f32 prefill
// attention (G24). Read here only; every consumer receives it as an explicit
// argument so no path can pick it up by accident — see runLayersFromEmbedN.
//
// Enabling it gives up two of acc64's three guarantees for this model:
// spec-decode verify == sequential greedy (structurally prevented from applying
// there anyway), and decode == prefill. Measured divergence at dense 1.5B:
// cosine ~0.9976, and a measured 2.28x on an 8k prefill
// (docs/measurements/attention-a3-kernel-ratio-2026-08-26.md).
func cpuFastAttention() bool { return os.Getenv("GOINFER_CPU_FAST_ATTENTION") == "1" }

// attnHeadsParThreshold gates A1 move (a)'s head-parallel fan-out: below this
// many MACs' worth of per-call work (K*nKeys, the QKᵀ/AV size driver), the
// fork-join cost isn't worth it and attendBatchedHeads runs its heads serially
// through pool[0] instead — the same "small work stays serial" discipline
// int4ParThreshold and aikit's own parThreshold already apply one level down
// (docs/task-attention-decode-cost.md's Gate A0 item 2 finding: a second
// confirmed instance of that bug class, here avoided rather than repeated).
// Measured, not guessed — see the campaign doc's move (a) writeup.
const attnHeadsParThreshold = 0

// canBatchN reports whether the batched M=K path applies: the gated-MLP families
// (Qwen / Llama / Gemma) AND standard sparse-MoE (Mellum / Mixtral) with K>1 —
// their attention is plain GQA softmax, so the SIMD attendBatchedHeads applies
// (the L² hotspot: a profile put scalar attendQuery at ~83% of MoE prefill). The
// MoE FFN itself stays per-row (router picks different experts per token).
// GPT-2 (non-gated + learned positions) and K≤1 take the sequential fallback.
func (m *Model) canBatchN(K int) bool {
	a := m.w.arch
	// Gemma 4 (per-layer head_dim, KV-sharing, PLE), qwen3_5_moe (Gated DeltaNet),
	// granitemoehybrid + nemotron_h (Mamba-2), and deepseek_v2/v3 (MLA latent cache)
	// each have their own sequential forward — the recurrent / latent-reconstruction
	// layers can't be batched across positions; exclude them all.
	return K > 1 && m.w.Embed.Rows() != 0 && !a.NonGatedMLP && !a.LearnedPosEmbed && a.gemma4 == nil && a.qwen35 == nil && a.granite == nil && a.nemotron == nil && a.mla == nil && a.llama4 == nil && a.gptoss == nil
}

// specRollbackSafe reports whether speculative decode's rollback — KVCache.TruncateTo
// after a partial accept — correctly restores this model's state. True for softmax /
// GQA (truncate the appended K/V) and MLA (reslice the latent KV) — both live in the
// cache, so a verified-then-rejected draft block leaves no residue. FALSE for the
// recurrent families — Mamba-2 (granite / nemotron_h) and Gated DeltaNet
// (qwen3_5_moe) — whose rolling state mamba2Step / the delta scan mutate IN PLACE and
// TruncateTo does NOT roll back (it only reslices KV layouts). Verifying a K-token
// block over-advances that state, and the next round decodes from it: a silent
// distribution bug, not a crash (00-core §6). Those families need the
// checkpoint-at-block-start / restore path (not yet built); until then the n-gram
// speculative entry points refuse them and the caller falls back to plain decode.
func (m *Model) specRollbackSafe() bool {
	a := m.w.arch
	if a.granite != nil || a.nemotron != nil || a.qwen35 != nil {
		return false
	}
	// C1/C-04: a STAGED sliding-window cache stores local layers in physical rings. Once a ring
	// wraps (context > window), a rollback of >1 position can't restore the evicted positions, so
	// the verify reads stale history and diverges — the "lossless" guarantee broken for the families
	// rings serve (Gemma-3 local / Mistral / Phi-3). The earlier exemption keyed on m.resident==nil,
	// assuming a resident backend means the positional resident path is taken — but three of four
	// speculative loops (EAGLE, grammar always; n-gram whenever a Session drives the staged cache)
	// use the staged ring EVEN when m.resident!=nil, and a resident CAS loss also falls back to
	// staged mid-flight. That misjudged path made the predicate return "safe" for a cache that wraps.
	// Refuse windowed models for speculation unconditionally: the resident positional path is itself
	// safe, but it is not a shipping speculative combo (CUDA declines windowing; Metal has no
	// speculation), so conservative refusal (→ plain decode) costs no real throughput and closes the
	// hole at the source. With windowed models refused here, the staged rollback sites
	// (KVCache.TruncateTo) only ever run on ring-free caches, where TruncateTo is always exact — so
	// no inexact case reaches them; re-enabling windowed speculation later must consume that exact
	// bool at each rollback site (audit C-04).
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
	// maxQDim, not qDim: a family with PER-LAYER query heads (Laguna — 48 on
	// full-attention layers, 64 on sliding) reuses q/ctx across every layer, so they
	// must be sized for the WIDEST one and sliced per layer below. maxHeads collapses
	// to NumHeads everywhere else, leaving these allocations unchanged.
	maxQDim, kvDim, inter := arch.maxHeads()*hd, nKV*hd, arch.IntermediateDim
	K := len(h) / hidden
	startPos := cache.Pos()
	sandwich := arch.NormPlacement == NormSandwich4
	parallel := arch.NormPlacement == NormParallel

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
	// A3 (G24): OPT-IN f32 attention for prefill. Off unless asked for, and
	// never for MoE.
	//
	// The acc64 path is 8.14x slower than f32 at long-context shapes (measured,
	// docs/measurements/attention-a3-kernel-ratio-2026-08-26.md — note that is
	// more than double the "~3.7x" the kernel comment assumes), and attention is
	// ~70% of an 8k prefill. It exists to hold three guarantees, and enabling this
	// gives up two of them for the model that enables it: spec-decode verify ==
	// sequential greedy, and decode == prefill. The third — MoE router stability —
	// is NOT negotiable at any flag setting, because an f32 QK reassociation flips
	// a top-k expert at a near-tie and cascades, so MoE is excluded here rather
	// than trusted to the operator.
	//
	// Same shape as --metal-fast-prefill: default off, divergence documented, and
	// the caller opts in knowingly.
	// fastAttn is the CALLER's statement that this sweep may diverge. It is a
	// parameter rather than a global read because the guard has to be structural:
	// spec-decode verify runs through forwardN and MUST keep acc64, or "verify ==
	// sequential greedy" silently stops holding. A runtime check could not tell
	// the two callers apart; a parameter cannot get it wrong.
	useAcc64 := !(fastAttn && arch.MoE == nil)

	// G16: prefill attention runs its heads in PARALLEL, budget permitting.
	//
	// A1 deferred this ("no M>1-specific work here") and implemented the deferral
	// literally, as one pool slot — which forced attendBatchedHeads's serial
	// branch below. The deferral had a measured cost: CPU prefill sat at ~100% of
	// one core on a 6-P-core box while the weight matmuls beside it fanned out,
	// and since serial attention is O(K²) while those matmuls are O(K), attention
	// took a growing share as prompts got longer.
	//
	// Nothing about the guarantee changes. A1's constraint permits exactly this —
	// "Parallelism may only split independent outputs across workers/registers —
	// heads, ..." — and attendBatchedHeads's own comment records that the nH query
	// heads are fully independent (disjoint ctx writes, no shared mutable state).
	// Each worker owns its own scratch slot. Bit-identity is gated by
	// TestPrefillAttnPoolInvariance, not assumed.
	//
	// The pool is BUDGETED, not simply maxAttnWorkers: a slot's scores buffer is
	// K*nKeys floats, quadratic in prompt length, so the worker count falls back
	// toward serial on long prompts rather than the allocation growing without
	// bound (prefillAttnWorkers).
	attnPool := newHeadWorkerPool(prefillAttnWorkers(K, maxKeys, hd, arch.maxHeads()), K, maxKeys, hd)
	// f32 scratch for the assembled local window (ring history + new rows) AND for
	// dequantizing int8 layers into for the f32 attention; ≤ maxKeys rows wide.
	// Allocated when the model has ring layers or an int8 cache.
	var alk, alv []float32
	if cache.localAny || cache.quant == kvI8 {
		alk = make([]float32, maxKeys*kvDim)
		alv = make([]float32, maxKeys*kvDim)
	}
	// Batch the qkv and gate/up projections (shared activation) so a GPU backend
	// runs each group as one submit (BatchTiled) instead of per-matmul syncs.
	var ws linalg.Workspace
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
		// G18: an abandoned client must not leave this loop running. Prefill is where
		// the time goes (a 3k-token prompt is minutes), and before this check a client
		// that gave up left a core burning to completion — measured at 47:38 of CPU
		// with nothing attached, with a retrying harness stacking one such prefill per
		// retry. Checked per LAYER, not per token: the check is free at this
		// granularity, but it is NOT instant — the bound is one layer's work, measured
		// at ~12s for a 3072-token prompt on an M1 Pro (cancel at 300ms, observed at
		// 12.34s; TestPrefillCancelMidFlight logs both). That is the tail to tighten
		// if it ever matters — per-head inside attendBatchedHeads — not a claim that
		// cancellation is immediate here.
		if err := reqCtx.Err(); err != nil {
			return nil, err
		}
		if m.layerPager != nil {
			m.layerPager.enterLayer(l) // dense weight streaming (#4)
		}
		lw := &m.w.Layers[l]
		global := arch.isGlobalLayer(l)
		// This layer's query width. q/ctx are sliced to it so every row stride, RoPE
		// call and matmul below sees the layer's OWN head count — the batched twin of
		// what causalAttention does for K=1.
		nH := arch.headsAt(l)
		qDim := nH * hd
		q, ctx := q[:K*qDim], ctx[:K*qDim]

		copy(norm, h)
		for i := range K {
			normalize(arch, row(norm, i, hidden), lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
		}
		if isW8A8(&lw.QProj) && isW8A8(&lw.KProj) && isW8A8(&lw.VProj) {
			qkvOps[0] = linalg.W8A8Op{BQ: wmInt8(&lw.QProj), Scales: wmScales(&lw.QProj), Dst: q, N: lw.QProj.Rows()}
			qkvOps[1] = linalg.W8A8Op{BQ: wmInt8(&lw.KProj), Scales: wmScales(&lw.KProj), Dst: k, N: lw.KProj.Rows()}
			qkvOps[2] = linalg.W8A8Op{BQ: wmInt8(&lw.VProj), Scales: wmScales(&lw.VProj), Dst: v, N: lw.VProj.Rows()}
			matmulW8A8Batch(be, &ws, norm, K, lw.QProj.Cols(), qkvOps[:])
		} else {
			matmul(be, &lw.QProj, norm, q, K)
			matmul(be, &lw.KProj, norm, k, K)
			matmul(be, &lw.VProj, norm, v, K)
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
				rmsNorm(qi, lw.QNorm, nH, hd, arch.NormEps, arch.RMSAddOne)
				rmsNorm(ki, lw.KNorm, nKV, hd, arch.NormEps, arch.RMSAddOne)
			}
			if !noPE {
				ropeAt(qi, nH, hd, pos, invFreq, ms, arch.MRopeSection, cache.mropePos, cache.mropeDelta, arch.ropeInterleave)
				ropeAt(ki, nKV, hd, pos, invFreq, ms, arch.MRopeSection, cache.mropePos, cache.mropeDelta, arch.ropeInterleave)
			}
			if !isLocal {
				cache.Append(l, ki, vi) // global: append now; local: deferred to commitBatch below
			}
		}
		// QKᵀ and scores·V for all K positions, per head, on the SIMD A·Bᵀ kernel
		// (the L² terms) instead of the scalar per-position attendQuery. f64
		// accumulation (the `true` acc64 arg) is bit-identical to the sequential
		// reference decode also runs (causalAttention), so a batched verify reproduces
		// sequential greedy EXACTLY — required for same-model speculative decoding and
		// for the MoE top-k router never to cascade. (Was f32 for dense — only cosine
		// ≥0.99, which broke spec parity since f32's reduction is M-dependent.)
		// Local layers read an assembled [base, startPos+K) window (ring history +
		// the K new rows in k/v); the ring write is deferred until after the read so
		// a K>W batch can't evict in-batch history. Global layers read append-forever.
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
		// Laguna output gating, per row, BEFORE o_proj — the batched twin of the K=1
		// call in causalAttention, sharing applyGateRow so the two cannot diverge.
		// `norm` still holds this layer's POST-input_layernorm rows here (it is not
		// recomputed for the MLP until after the o_proj below), which is exactly the
		// tensor the gate reads.
		if arch.laguna != nil {
			gRows := lw.GProj.Rows()
			if cap(gbuf) < K*gRows {
				gbuf = make([]float32, K*gRows)
			}
			gb := gbuf[:K*gRows]
			matmul(be, &lw.GProj, norm, gb, K)
			perHead := gRows == nH
			for i := range K {
				applyGateRow(row(gb, i, gRows), row(ctx, i, qDim), perHead, nH, hd)
			}
		}
		matmul(be, &lw.OProj, ctx, att, K)
		if arch.OutBias {
			for i := range K {
				addBias(row(att, i, hidden), lw.OBias)
			}
		}
		if sandwich {
			for i := range K {
				normalize(arch, row(att, i, hidden), lw.PostAttnNorm, nil, hidden)
			}
		}
		if !parallel {
			// Sequential: add the attention residual, then re-norm the updated stream for the MLP.
			for j := range h {
				h[j] += att[j]
			}
			copy(norm, h)
			for i := range K {
				normalize(arch, row(norm, i, hidden), lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
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
			// Sparse MoE (Mellum / Mixtral): the router selects different experts per
			// token, so the FFN isn't batchable across K — run the existing per-token
			// moeMLP for each row (bit-identical to the sequential path). The prefill
			// hotspot (attention) is already batched above; this is the residual ~17%.
			// GLM's dense prefix layers (Experts nil) fall through to the dense FFN below.
			for i := range K {
				// nil scr: this batched-prefill path builds its own per-K-batch scratch
				// above and has no cache.scr in scope; moeMLP falls back to allocating,
				// amortized over K tokens (not the flagged single-token decode hot path).
				ff, err := moeMLP(row(norm, i, hidden), lw, arch, be, nil, m.pager)
				if err != nil {
					return nil, err
				}
				if sandwich {
					normalize(arch, ff, lw.PostMLPNorm, nil, hidden)
				}
				hi := row(h, i, hidden)
				for j := range ff {
					hi[j] += ff[j]
				}
			}
			// Hidden-state seam (05), same as the dense path below: MoE layers must also
			// record captured[ci], or GenerateEagleSpeculative against a sparse-MoE target
			// (Mixtral/Mellum) leaves captured all-nil and fuseAt slices a nil slice → panic
			// (audit C-07). The `continue` used to skip this.
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
			matmulW8A8Batch(be, &ws, norm, K, lw.GateProj.Cols(), guOps[:])
		} else {
			matmul(be, &lw.GateProj, norm, gate, K)
			matmul(be, &lw.UpProj, norm, up, K)
		}
		switch arch.Act {
		case ActGeluTanh:
			for j := range gate {
				gate[j] = geluTanh(gate[j]) * up[j]
			}
		case ActSiLU:
			for j := range gate {
				gate[j] = silu(gate[j]) * up[j]
			}
		default:
			return nil, errNotImplemented
		}
		matmul(be, &lw.DownProj, gate, mlpOut, K)
		if sandwich {
			for i := range K {
				normalize(arch, row(mlpOut, i, hidden), lw.PostMLPNorm, nil, hidden)
			}
		}
		if parallel {
			// Single residual add: attention + MLP, both from the shared input norm.
			for j := range h {
				h[j] += att[j] + mlpOut[j]
			}
		} else {
			for j := range h {
				h[j] += mlpOut[j]
			}
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

// attendBatchedHeads computes grouped-query causal attention for K query
// positions at once, per head, via the SIMD A·Bᵀ matmul (linalg.MatmulBT)
// instead of the scalar per-position attendQuery. The two O(L²) terms — QKᵀ and
// scores·V — move off the scalar triple-loops onto the vector kernel, which an
// end-to-end prefill profile showed were ~half the forward's CPU time.
//
// Per KV head it gathers K_head [nKeys,hd] and V_headᵀ [hd,nKeys] once (reused
// across the GQA group). Per query head: scores[K,nKeys] = Q_head·K_headᵀ; a
// scaled, causal/window-masked softmax per row (row i attends to
// [WindowStart(startPos+i), startPos+i], masked entries zeroed so they drop out
// of the next matmul); then ctx_head[K,hd] = scores·V_head, expressed as
// MatmulBT(scores, V_headᵀ); scattered into ctx[K,qDim].
//
// NOT bit-identical to attendQuery: QKᵀ moves from float64 to f32 accumulation
// and the matmul reassociates the reduction. Parity is argmax-exact + cosine —
// the same standard the GPU residency attention already meets. The softmax exp
// stays per-row in float64. Scratch slices (qh:[K*hd], kh:[maxKeys*hd],
// vt:[maxKeys*hd], scores:[K*maxKeys], ch:[K*hd]) are caller-owned, reused across
// layers.
// keys/vals are the contiguous K/V the gather reads, with physical row 0 holding
// absolute key position `base`: for a global layer that's cache.Keys(layer) at
// base 0; for a local (sliding-window) layer it's an assembled [base, startPos+K)
// window (the resident ring history + the K new rows) so the ring's wrap is
// invisible here and the math is byte-identical to append-forever. Per-query
// masking stays in absolute positions (WindowStart/attendHi) and maps to physical
// columns s-base.
func attendBatchedHeads(q, ctx, keys, vals []float32, base int, cache *KVCache, layer, startPos, K int, global bool, arch *Architecture, useAcc64 bool, pool []headWorkerScratch) {
	// headsAt, not NumHeads: Laguna varies the QUERY head count per layer (its KV
	// heads stay uniform, so `group` below is per-layer too). Every other family's
	// headsAt returns NumHeads, leaving this identical.
	nH, nKV, hd := arch.headsAt(layer), arch.NumKVHeads, arch.HeadDim
	kvDim, qDim := nKV*hd, nH*hd
	group := nH / nKV
	scale := arch.AttnScale
	nKeys := len(keys) / kvDim
	// MoE routing is discontinuous: the f32 QKᵀ reassociation (~4.6e-5) flips a
	// top-k expert at a near-tie and cascades, changing the output. MatmulBTAcc64
	// accumulates the dot in f64 (bit-identical to the sequential f64 reference),
	// killing that perturbation — ~3.7× slower than f32 but still ≫ the scalar
	// path. Dense MLPs tolerate the f32 error (cosine ≥0.99), so they keep MatmulBT.
	matmul := linalg.MatmulBT
	if useAcc64 {
		matmul = linalg.MatmulBTAcc64
	}

	// attendOneHead runs one query head's QKᵀ → softmax → scores·V → scatter into
	// ctx, using ws's scratch. A1 move (a): this is what runs concurrently across
	// heads below — bit-identical regardless of which pool slot or goroutine runs
	// it, or what order heads finish in, since every head's own math (moves b/c's
	// unchanged per-output reduction order) and its ctx write (a disjoint qhead*hd
	// slice — no two heads ever touch the same bytes) are exactly as before.
	attendOneHead := func(qhead int, ws *headWorkerScratch) {
		kvh := qhead / group
		qh, scores, ch, avAcc := ws.qh, ws.scores, ws.ch, ws.avAcc
		// G20: walk the query rows in TILES. Every step below is row-wise — the Q
		// gather, QKᵀ (rows are the leading dimension), the per-row softmax, scores·V
		// (each row folds over keys independently) and the ctx scatter — so splitting
		// rows splits INDEPENDENT OUTPUTS, which is what A1's bit-identity constraint
		// permits. No key-dimension split happens here and none may: that would
		// re-associate the softmax denominator and the AV fold, the exact thing acc64
		// exists to prevent.
		//
		// The point is memory, not speed: scores is tile*nKeys instead of K*nKeys, so
		// a worker slot stops growing with the square of the prompt and the G16 pool
		// can still fan out on a long prompt.
		//
		// `i` indexes the TILE below; `gi` is the global row. Positions and masks must
		// use `gi` — startPos+gi, treeRowPos[gi], treeMask[gi] — while buffers use `i`.
		tile := attnRowTile(K, nKeys)
		for t0 := 0; t0 < K; t0 += tile {
			kt := min(tile, K-t0)
			for i := range kt { // gather this tile's Q_head [kt,hd]
				b := (t0+i)*qDim + qhead*hd
				copy(qh[i*hd:i*hd+hd], q[b:b+hd])
			}
			// QKᵀ: scores[K,nKeys] = Q_head[K,hd] · K_head[nKeys,hd]ᵀ. Acc64 reads
			// keys DIRECTLY — row stride kvDim (rows are nKeys apart), element
			// stride 1 (a head's hd floats are contiguous) — skipping a kh gather
			// entirely. Bit-identical by construction (P1; aikit v1.18.0
			// MatmulBTAcc64Strided runs the SAME sequential f64 reduction as
			// MatmulBTAcc64, only b's addressing differs), verified at goinfer's own
			// stride parameters by TestAttendStrided_matchesGatherReference.
			//
			// A1 move (b): MatmulQKAcc64 interleaves 8 keys' dot products as 8
			// concurrent f64 accumulator chains, hiding FMA latency the single-chain
			// dotF32Acc64 leaves idle (each key's own d-order fold is unchanged, so
			// this is bit-identical, not just close — docs/task-attention-decode-cost.md).
			// Measured 4.4x in isolation (both depth 130 and 8192 — a pure latency
			// fix, not depth-dependent, unlike move (c)'s memory-order fix).
			if useAcc64 {
				linalg.MatmulQKAcc64(qh[:kt*hd], keys, scores[:kt*nKeys], kt, hd, nKeys, kvh*hd, kvDim)
			} else {
				matmul(qh[:kt*hd], ws.kh[:nKeys*hd], scores[:kt*nKeys], kt, hd, nKeys)
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
							sc := float64(rowS[s]) * scale
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
				maxS := math.Inf(-1)
				for s := loP; s <= hiP; s++ {
					sc := float64(rowS[s]) * scale
					rowS[s] = float32(sc)
					if sc > maxS {
						maxS = sc
					}
				}
				var sum float64
				for s := loP; s <= hiP; s++ {
					e := math.Exp(float64(rowS[s]) - maxS)
					rowS[s] = float32(e)
					sum += e
				}
				inv := 1.0 / sum
				for s := range loP {
					rowS[s] = 0
				}
				for s := loP; s <= hiP; s++ {
					rowS[s] = float32(float64(rowS[s]) * inv)
				}
				for s := hiP + 1; s < nKeys; s++ {
					rowS[s] = 0
				}
			}
			// scores·V: ctx_head[K,hd] = scores[K,nKeys] · V_head[nKeys,hd]
			//                          = MatmulBT(scores, V_headᵀ[hd,nKeys])
			// Acc64 reads vals DIRECTLY, "as if transposed" — row stride 1 (V's hd
			// floats are contiguous, and vt's row index IS that offset), element
			// stride kvDim (vt's column index steps by a whole KV row) — skipping
			// a vt gather+transpose. Same bit-identity argument as QKᵀ.
			//
			// A1 move (c): MatmulAVAcc64 reads V rows contiguously (keys-outer,
			// dims-inner) into hd independent f64 accumulators, instead of
			// MatmulBTAcc64Strided's dims-outer/keys-inner walk (one cache line
			// per f64 MAC at kvDim stride). Bit-identical by construction — each
			// dim's accumulator sees the same key-ascending sequence of adds
			// either way (docs/task-attention-decode-cost.md, docs/task-decode-
			// splitkv-attention.md:36's "split the independent axis" principle).
			// Measured 1.81x at depth 130, 2.39x at depth 8192 (aikit
			// MatmulAVAcc64_ABBench).
			if useAcc64 {
				linalg.MatmulAVAcc64(scores[:kt*nKeys], vals, ch[:kt*hd], avAcc, kt, nKeys, hd, kvh*hd, kvDim)
			} else {
				matmul(scores[:kt*nKeys], ws.vt[:hd*nKeys], ch[:kt*hd], kt, nKeys, hd)
			}
			for i := range kt { // scatter this tile's ctx_head into ctx[K,qDim]
				b := (t0+i)*qDim + qhead*hd
				copy(ctx[b:b+hd], ch[i*hd:i*hd+hd])
			}
		}
	}

	if !useAcc64 {
		// f32 fallback: test-only in practice (every live caller passes
		// useAcc64=true — see the comment above). Runs through pool[0] alone,
		// with the per-kvh kh/vt gather ORIGINAL attendBatchedHeads did before
		// A1 move (a) — every query head in a kv group still reuses the SAME
		// gathered kh/vt, so this stays single-threaded (the gather itself is
		// shared, mutable state a concurrent split would race on).
		ws := &pool[0]
		for kvh := range nKV {
			for s := range nKeys {
				kvBase := s*kvDim + kvh*hd
				copy(ws.kh[s*hd:s*hd+hd], keys[kvBase:kvBase+hd])
				vrow := vals[kvBase : kvBase+hd]
				for d := range hd {
					ws.vt[d*nKeys+s] = vrow[d]
				}
			}
			for g := range group {
				attendOneHead(kvh*group+g, ws)
			}
		}
		return
	}

	// A1 move (a): the acc64 path (the real one — every live caller). The nH
	// query heads are fully independent (disjoint ctx writes, no shared mutable
	// state — kh/vt aren't touched on this path at all). Below the measured
	// fan-out floor, or with only one pool slot / one head, run serially through
	// pool[0] instead: a fork-join here costs real time the (c)+(b) speedups
	// already shrank to ~13-14 µs/head at depth 130 — the same "small work stays
	// serial" discipline int4ParThreshold and aikit's parThreshold apply one
	// level down (Gate A0 item 2 — this is the SAME bug class, avoided here
	// rather than repeated a third time).
	if len(pool) <= 1 || nH <= 1 || K*nKeys < attnHeadsParThreshold {
		ws := &pool[0]
		for qhead := range nH {
			attendOneHead(qhead, ws)
		}
		return
	}
	workers := min(len(pool), nH)
	var wg sync.WaitGroup
	headsPer := (nH + workers - 1) / workers
	for w := range workers {
		h0, h1 := w*headsPer, min((w+1)*headsPer, nH)
		if h0 >= h1 {
			continue
		}
		wg.Add(1)
		go func(w, h0, h1 int) {
			defer wg.Done()
			ws := &pool[w]
			for qhead := h0; qhead < h1; qhead++ {
				attendOneHead(qhead, ws)
			}
		}(w, h0, h1)
	}
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
	// logit_scale (Cohere multiplier stored as goinfer's reciprocal; Granite
	// logits_scaling divisor). Mirror logitsFromHidden's tail — the sequential
	// path applies this, so the batched prefill/verify must too, else forwardN
	// diverges for any LogitScale family. Cohere is the first forwardN-eligible
	// one (Granite runs its own forward), which is why this was latent.
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
	// Compute-time LoRA (#7) is wired only into the sequential forward, so an active
	// adapter must take the M=1 path (as prefillLogits does): the batched verify would
	// project every position with the base model and commit base K/V, silently
	// verifying speculative drafts against the wrong model (M12).
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
	// forwardN backs speculative verify: never fast, whatever the operator asked for.
	h, err := m.forwardLayersN(reqCtx, ids, cache, false)
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

// prefillLogits processes the whole prompt and returns the logits at its LAST
// position (the seed for the first generated token). On the batched archs it
// runs the layers at M=len(prompt) in one pass — each weight streamed once,
// reused across all positions (~1.7–2× faster prompt prefill / time-to-first-
// token than sequential M=1) — and runs the LM head on the last position ONLY
// (the others' logits aren't needed). Falls back to sequential runLayers +
// forward otherwise. Bit-identical to the sequential prefill (the seed token is
// unchanged). The cache is filled with the whole prompt either way.
func (m *Model) prefillLogits(ctx context.Context, prompt []int, cache *KVCache) ([]float32, error) {
	// Compute-time LoRA (#7) is wired only into the sequential forward (causalAttention
	// + gatedMLP), so an active adapter takes the M=1 path — the prompt's K/V must carry
	// the delta or decode would continue a base-projected context. The RAM-density win
	// (N adapters share one base) is unaffected; only adapter'd prefill speed regresses.
	if cache.lora != nil || !m.canBatchN(len(prompt)) {
		for _, id := range prompt[:len(prompt)-1] {
			// G18: the sequential fallback checks per token — it has no layer batch to
			// bound, and a LoRA'd or non-batchable arch prefills here.
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if _, err := m.runLayers(id, cache); err != nil {
				return nil, err
			}
		}
		return m.forward(prompt[len(prompt)-1], cache)
	}
	h, err := m.forwardLayersN(ctx, prompt, cache, cpuFastAttention())
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
	if !m.canBatchN(len(ids)) {
		return nil, fmt.Errorf("decoder: multimodal prefill needs the batched path (canBatchN false)")
	}
	hidden := m.w.arch.HiddenDim
	if imgPos < 0 || imgLen <= 0 || imgPos+imgLen > len(ids) {
		return nil, fmt.Errorf("decoder: image run [%d,%d) out of range for %d tokens", imgPos, imgPos+imgLen, len(ids))
	}
	if len(imageEmbeds) != imgLen*hidden {
		return nil, fmt.Errorf("decoder: imageEmbeds len %d, want %d (%d tokens × %d)", len(imageEmbeds), imgLen*hidden, imgLen, hidden)
	}
	h := m.embedN(ids)
	copy(h[imgPos*hidden:(imgPos+imgLen)*hidden], imageEmbeds) // raw projected features, no embed scale
	cache.SetImageBlocks([][2]int{{imgPos, imgPos + imgLen}})
	hN, err := m.runLayersFromEmbedN(ctx, h, cache, cpuFastAttention())
	if err != nil {
		return nil, err
	}
	return m.lmHeadN(hN[(len(ids)-1)*hidden:], 1), nil
}

// prefillLogitsQwenVL prefills a Qwen2.5-VL multimodal prompt and returns the
// last-position logits: text embeddings with the merged vision features
// (imageFeats, [imgLen*HiddenDim] from the ViT+merger) substituted at the <image>
// run [imgPos,imgPos+imgLen), under m-RoPE 3D positions (mropePos, one (t,h,w) per
// absolute sequence position). Two deltas vs prefillLogitsVL (Gemma 3): the image
// tokens attend CAUSALLY — no bidirectional image block (Qwen's bidirectionality is
// inside the ViT, not the decoder) — and the rotary uses m-RoPE (ropeAt reads
// cache.mropePos). The merged features are RAW (no embed scale), matching HF's
// scatter into inputs_embeds. (P5)
func (m *Model) prefillLogitsQwenVL(ctx context.Context, ids []int, imageFeats []float32, imgPos, imgLen int, mropePos [][3]int, cache *KVCache) ([]float32, error) {
	if !m.canBatchN(len(ids)) {
		return nil, fmt.Errorf("decoder: Qwen2.5-VL prefill needs the batched path (canBatchN false)")
	}
	hidden := m.w.arch.HiddenDim
	if imgPos < 0 || imgLen <= 0 || imgPos+imgLen > len(ids) {
		return nil, fmt.Errorf("decoder: image run [%d,%d) out of range for %d tokens", imgPos, imgPos+imgLen, len(ids))
	}
	if len(imageFeats) != imgLen*hidden {
		return nil, fmt.Errorf("decoder: imageFeats len %d, want %d (%d tokens × %d)", len(imageFeats), imgLen*hidden, imgLen, hidden)
	}
	if len(mropePos) != len(ids) {
		return nil, fmt.Errorf("decoder: mropePos len %d, want %d (one per token)", len(mropePos), len(ids))
	}
	h := m.embedN(ids)
	copy(h[imgPos*hidden:(imgPos+imgLen)*hidden], imageFeats) // raw merged features, no embed scale
	cache.mropePos = mropePos                                 // ropeAt switches to m-RoPE for this prefill
	cache.mropeDelta = mropeDelta(mropePos, len(ids))         // decode past the prefill rotates at seqPos+delta
	hN, err := m.runLayersFromEmbedN(ctx, h, cache, cpuFastAttention())
	if err != nil {
		return nil, err
	}
	return m.lmHeadN(hN[(len(ids)-1)*hidden:], 1), nil
}

package decoder

import (
	"fmt"
	"math"

	"github.com/townsendmerino/aikit/linalg"
)

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
func (m *Model) forwardLayersN(ids []int, cache *KVCache) ([]float32, error) {
	return m.runLayersFromEmbedN(m.embedN(ids), cache)
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
func (m *Model) runLayersFromEmbedN(h []float32, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	be := m.be
	hidden, nH, nKV, hd := arch.HiddenDim, arch.NumHeads, arch.NumKVHeads, arch.HeadDim
	qDim, kvDim, inter := nH*hd, nKV*hd, arch.IntermediateDim
	K := len(h) / hidden
	startPos := cache.Pos()
	sandwich := arch.NormPlacement == NormSandwich4
	parallel := arch.NormPlacement == NormParallel

	norm := make([]float32, K*hidden)
	q := make([]float32, K*qDim)
	k := make([]float32, K*kvDim)
	v := make([]float32, K*kvDim)
	ctx := make([]float32, K*qDim)
	att := make([]float32, K*hidden)
	gate := make([]float32, K*inter)
	up := make([]float32, K*inter)
	mlpOut := make([]float32, K*hidden)
	// Batched per-head attention scratch (reused across layers; nKeys = startPos+K
	// is the same for every layer in this sweep). See attendBatchedHeads.
	maxKeys := startPos + K
	aqh := make([]float32, K*hd)
	akh := make([]float32, maxKeys*hd)
	avt := make([]float32, maxKeys*hd)
	ascores := make([]float32, K*maxKeys)
	ach := make([]float32, K*hd)
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

	row := func(b []float32, i, w int) []float32 { return b[i*w : i*w+w] }

	if m.layerPager != nil {
		defer m.layerPager.finishLayers()
	}
	for l := 0; l < arch.NumLayers; l++ {
		if m.layerPager != nil {
			m.layerPager.enterLayer(l) // dense weight streaming (#4)
		}
		lw := &m.w.Layers[l]
		global := arch.isGlobalLayer(l)

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
			attendBatchedHeads(q, ctx, alk[:nRows*kvDim], alv[:nRows*kvDim], base, cache, l, startPos, K, global, arch, true, aqh, akh, avt, ascores, ach)
			cache.commitBatch(l, startPos, K, k, v)
		} else if cache.quant == kvI8 {
			// int8 global: the Append loop above already quantized the new K/V into
			// the layer; dequant the full history into f32 scratch for the matmul.
			nKeys := cache.dequantGlobalLayer(l, kvDim, alk, alv)
			attendBatchedHeads(q, ctx, alk[:nKeys*kvDim], alv[:nKeys*kvDim], 0, cache, l, startPos, K, global, arch, true, aqh, akh, avt, ascores, ach)
		} else {
			attendBatchedHeads(q, ctx, cache.Keys(l), cache.Vals(l), 0, cache, l, startPos, K, global, arch, true, aqh, akh, avt, ascores, ach)
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
				ff, err := moeMLP(row(norm, i, hidden), lw, arch, be, m.pager)
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
func attendBatchedHeads(q, ctx, keys, vals []float32, base int, cache *KVCache, layer, startPos, K int, global bool, arch *Architecture, useAcc64 bool, qh, kh, vt, scores, ch []float32) {
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

	for kvh := range nKV {
		if !useAcc64 {
			// Gather this KV head's keys [nKeys,hd] and values transposed [hd,nKeys]
			// once; every query head in the GQA group reuses them. The V transpose is
			// folded into the gather (free) so scores·V is MatmulBT(scores, V_headᵀ).
			// (Acc64 path below reads keys/vals directly instead — P1. Every live
			// caller passes useAcc64=true, so this gather only runs under the f32
			// fallback exercised by tests; aikit has not shipped a bit-identical
			// strided kernel for that reassociating path.)
			for s := range nKeys {
				kvBase := s*kvDim + kvh*hd
				copy(kh[s*hd:s*hd+hd], keys[kvBase:kvBase+hd])
				vrow := vals[kvBase : kvBase+hd]
				for d := range hd {
					vt[d*nKeys+s] = vrow[d]
				}
			}
		}
		for g := range group {
			qhead := kvh*group + g
			for i := range K { // gather Q_head [K,hd]
				base := i*qDim + qhead*hd
				copy(qh[i*hd:i*hd+hd], q[base:base+hd])
			}
			// QKᵀ: scores[K,nKeys] = Q_head[K,hd] · K_head[nKeys,hd]ᵀ. Acc64 reads
			// keys DIRECTLY — row stride kvDim (rows are nKeys apart), element
			// stride 1 (a head's hd floats are contiguous) — skipping the kh gather
			// above entirely. Bit-identical by construction (P1; aikit v1.18.0
			// MatmulBTAcc64Strided runs the SAME sequential f64 reduction as
			// MatmulBTAcc64, only b's addressing differs), verified at goinfer's own
			// stride parameters by TestAttendStrided_matchesGatherReference.
			if useAcc64 {
				linalg.MatmulBTAcc64Strided(qh[:K*hd], keys, scores[:K*nKeys], K, hd, nKeys, kvh*hd, kvDim, 1)
			} else {
				matmul(qh[:K*hd], kh[:nKeys*hd], scores[:K*nKeys], K, hd, nKeys)
			}
			// Scaled, masked softmax per query row; zero the out-of-range entries
			// so they contribute nothing to the scores·V matmul below.
			for i := range K {
				pos := startPos + i
				if cache.treeRowPos != nil {
					pos = cache.treeRowPos[i]
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
						return j < K && cache.treeMask[i][j]
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
			// the vt gather+transpose above. Same bit-identity argument as QKᵀ.
			if useAcc64 {
				linalg.MatmulBTAcc64Strided(scores[:K*nKeys], vals, ch[:K*hd], K, nKeys, hd, kvh*hd, 1, kvDim)
			} else {
				matmul(scores[:K*nKeys], vt[:hd*nKeys], ch[:K*hd], K, nKeys, hd)
			}
			for i := range K { // scatter ctx_head into ctx[K,qDim]
				base := i*qDim + qhead*hd
				copy(ctx[base:base+hd], ch[i*hd:i*hd+hd])
			}
		}
	}
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
func (m *Model) forwardN(ids []int, cache *KVCache) ([][]float32, error) {
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
	h, err := m.forwardLayersN(ids, cache)
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
func (m *Model) prefillLogits(prompt []int, cache *KVCache) ([]float32, error) {
	// Compute-time LoRA (#7) is wired only into the sequential forward (causalAttention
	// + gatedMLP), so an active adapter takes the M=1 path — the prompt's K/V must carry
	// the delta or decode would continue a base-projected context. The RAM-density win
	// (N adapters share one base) is unaffected; only adapter'd prefill speed regresses.
	if cache.lora != nil || !m.canBatchN(len(prompt)) {
		for _, id := range prompt[:len(prompt)-1] {
			if _, err := m.runLayers(id, cache); err != nil {
				return nil, err
			}
		}
		return m.forward(prompt[len(prompt)-1], cache)
	}
	h, err := m.forwardLayersN(prompt, cache)
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
func (m *Model) prefillLogitsVL(ids []int, imageEmbeds []float32, imgPos, imgLen int, cache *KVCache) ([]float32, error) {
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
	hN, err := m.runLayersFromEmbedN(h, cache)
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
func (m *Model) prefillLogitsQwenVL(ids []int, imageFeats []float32, imgPos, imgLen int, mropePos [][3]int, cache *KVCache) ([]float32, error) {
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
	hN, err := m.runLayersFromEmbedN(h, cache)
	if err != nil {
		return nil, err
	}
	return m.lmHeadN(hN[(len(ids)-1)*hidden:], 1), nil
}

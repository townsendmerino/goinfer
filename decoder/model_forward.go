package decoder

import (
	"context"
	"fmt"
	"math"
	"sync/atomic"
	"time"
)

// runLayers advances one decode step for token id at position cache.Pos(): it embeds the token, runs the block stack
// (appending this position's K/V to the cache), and returns the residual-stream hidden state after the final layer, before the
// final norm and LM head. Splitting it out lets prefill skip the (vocab-sized) LM head on every token but the last.
//
// The loop is generic over the Architecture descriptor: embedding scale, norm placement (Gemma's 4-norm sandwich vs Llama's
// pre-2), the (1+w) RMS offset, and the activation are all knobs. Gemma 3 is one descriptor:
//
//	h = Embed[id] * EmbedScale
//	for each layer l:
//	  n  = rmsNorm(h, PreAttnNorm)
//	  a  = causalAttention(l, n, ...)
//	  if Sandwich4 { a = rmsNorm(a, PostAttnNorm) }
//	  h += a
//	  n2 = rmsNorm(h, PreMLPNorm)
//	  g  = gatedMLP(n2, ...)
//	  if Sandwich4 { g = rmsNorm(g, PostMLPNorm) }
//	  h += g
func (m *Model) runLayers(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if m.w.Embed.Rows() == 0 {
		return nil, fmt.Errorf("decoder.forward: weights not loaded %w [M1]", errNotImplemented)
	}
	if cache.scr == nil { // caches from NewKVCache directly (tests); Generate uses NewCache
		cache.scr = newDecodeScratch(arch)
	}
	scr := cache.scr
	hidden := arch.HiddenDim
	h := scr.h           // residual stream (reused per stream; fully overwritten below)
	m.w.Embed.Row(id, h) // f32 copy, or int8 dequant when quantized
	// Embedding scale (Gemma = √hidden; 0/1 = none). NOTE: HF computes this
	// normalizer as sqrt(hidden) cast to the model's dtype — bf16 for a bf16
	// checkpoint (≈25.25 here) — then multiplies. We use the f32 value
	// (≈25.2982). It matches our parity gate because the next op (PreAttnNorm
	// RMSNorm) divides out a global scalar, so the difference only survives in
	// the residual and stays well under the ≥1−1e-4 cosine bar. If that bar is
	// ever tightened past ~1e-5, round the scale to bf16.
	emb := m.program().embed
	if emb.Scale != 0 && emb.Scale != 1 {
		scale := float32(emb.Scale)
		for i := range h {
			h[i] *= scale
		}
	}
	// Learned absolute position embedding (GPT-2): add wpe[pos], where pos is
	// this token's absolute position (the cache advances on Append inside
	// attention, so cache.Pos() here is still this step's position).
	if emb.LearnedPos {
		pe := make([]float32, hidden)
		m.w.PosEmbed.Row(cache.Pos(), pe)
		for i := range h {
			h[i] += pe[i]
		}
	}
	return m.runLayersFromEmbedPLE(h, id, cache)
}

// runLayersFromEmbed runs the transformer layers over a precomputed
// residual-stream embedding h ([hidden]) and returns it (mutated in place). For
// a text token h is the embedding lookup (+ scale/pos) that runLayers just
// computed; for an IMAGE position (multimodal) it is the projected vision
// embedding the interleaver substitutes in place of a token-id lookup. Splitting
// it out is the embed-by-vector seam: text behavior is unchanged (runLayers is
// exactly embed-then-this), and image embeddings reach the decoder without
// passing through embed_tokens. Generic path only (gemma4/qwen35 have their own
// runLayers and grow the same seam when those families go multimodal).
func (m *Model) runLayersFromEmbed(h []float32, cache *KVCache) ([]float32, error) {
	pad := -1
	if g := m.w.arch.gemma4; g != nil {
		pad = g.PadTokenID // a row with no token behind it takes the pad id's PLE, as HF's multimodal forward does
	}
	return m.runLayersFromEmbedPLE(h, pad, cache)
}

// runLayersFromEmbedPLE is runLayersFromEmbed with the token id behind h for Gemma 4's per-layer embedding: the id itself for a
// text row, the checkpoint's pad id for a multimodal row (docs/multimodal.md, P7 entry).
func (m *Model) runLayersFromEmbedPLE(h []float32, pleID int, cache *KVCache) ([]float32, error) {
	if cache.scr == nil { // direct callers (tests / the interleaver) may skip runLayers
		cache.scr = newDecodeScratch(m.w.arch)
	}
	if programWalkOffForTest {
		return m.runLayersFromEmbedLoop(h, cache)
	}
	return m.walkToken(m.program(), h, pleID, cache)
}

// runLayersFromEmbedLoop is the per-token loop the program walker replaced, kept for gate S's in-process A/B
// (programWalkOffForTest) until OP3's last step.
func (m *Model) runLayersFromEmbedLoop(h []float32, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	scr := cache.scr
	hidden := arch.HiddenDim
	if m.layerPager != nil {
		defer m.layerPager.finishLayers()
	}
	for l := 0; l < arch.NumLayers; l++ {
		if m.layerPager != nil {
			m.layerPager.enterLayer(l) // prefetch l+1, release the layer behind (#4)
		}
		lw := &m.w.Layers[l]
		var ld *loraLayerDelta // compute-time LoRA deltas for this layer (#7); nil when no adapter is active
		if cache.lora != nil {
			ld = &cache.lora.layers[l]
		}
		// Resolved per layer, not hoisted: NormPlacementLinear (Olmo Hybrid) can give
		// this layer a different placement than the rest of the model.
		placement := arch.normPlacementAt(l)
		sandwich := placement == NormSandwich4
		// postOnly (Olmo 3, and Olmo Hybrid's full-attention layers): no pre-norm at all —
		// the sublayer reads the RAW residual stream — but the sublayer's OUTPUT still gets
		// normalized before the residual add, the same as Sandwich4's post-norm half.
		// postNorm covers both placements that need it.
		postOnly := placement == NormPostOnly
		postNorm := sandwich || postOnly
		parallel := placement == NormParallel
		if parallel {
			// Cohere/GPT-J parallel block: ONE shared input norm feeds both
			// sublayers, whose outputs sum into a SINGLE residual add —
			// h += attn(n) + mlp(n), where n = norm(h). Both read the same n
			// (attention leaves scr.norm intact), so MLP takes scr.norm, NOT the
			// post-attention residual. No pre-MLP norm, no post-sublayer norms.
			normalizeInto(arch, scr.norm, h, lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
			if err := causalAttention(l, scr.norm, scr.sub, lw, arch, cache, m.be, ld); err != nil {
				return nil, err
			}
			if cache.subCapture { // scr.ctx is the pre-o-proj context; scr.sub is not yet overwritten
				cache.subCtx[l] = append(cache.subCtx[l][:0], scr.ctx...)
				cache.subAttn[l] = append(cache.subAttn[l][:0], scr.sub...)
			}
			if err := mlp(scr.norm, scr.sub2, lw, arch, m.be, scr, m.pager, ld); err != nil {
				return nil, err
			}
			if cache.subCapture { // no post-norm ⇒ the pre and final MLP contributions are identical
				cache.subMLPpre[l] = append(cache.subMLPpre[l][:0], scr.sub2...)
				cache.subMLP[l] = append(cache.subMLP[l][:0], scr.sub2...)
			}
			addResidual2(h, scr.sub, scr.sub2)
		} else {
			if !postOnly {
				normalizeInto(arch, scr.norm, h, lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
			} else {
				copy(scr.norm, h)
			}
			var dt0 time.Time
			if decodeTiming {
				dt0 = time.Now()
			}
			if err := causalAttention(l, scr.norm, scr.sub, lw, arch, cache, m.be, ld); err != nil {
				return nil, err
			}
			if decodeTiming {
				atomic.AddInt64(&dtAttn, int64(time.Since(dt0)))
			}
			if cache.subCapture { // scr.ctx is the pre-o-proj context; scr.sub is not yet overwritten
				cache.subCtx[l] = append(cache.subCtx[l][:0], scr.ctx...)
			}
			if postNorm {
				normalize(arch, scr.sub, lw.PostAttnNorm, nil, hidden)
			}
			if cache.subCapture { // scr.sub is now the attention contribution about to hit the residual
				cache.subAttn[l] = append(cache.subAttn[l][:0], scr.sub...)
			}
			addResidual(h, scr.sub)
			if !postOnly {
				normalizeInto(arch, scr.norm, h, lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
			} else {
				copy(scr.norm, h)
			}
			if decodeTiming {
				dt0 = time.Now()
			}
			if err := mlp(scr.norm, scr.sub, lw, arch, m.be, scr, m.pager, ld); err != nil {
				return nil, err
			}
			if decodeTiming {
				atomic.AddInt64(&dtMLP, int64(time.Since(dt0)))
			}
			if cache.subCapture { // scr.sub is the down output BEFORE the post-MLP sandwich norm
				cache.subMLPpre[l] = append(cache.subMLPpre[l][:0], scr.sub...)
			}
			if postNorm {
				normalize(arch, scr.sub, lw.PostMLPNorm, nil, hidden)
			}
			if cache.subCapture { // scr.sub is now the MLP contribution about to hit the residual
				cache.subMLP[l] = append(cache.subMLP[l][:0], scr.sub...)
			}
			addResidual(h, scr.sub)
		}
		// Read-only hidden-state seam (05): copy this layer's output residual stream
		// when requested. A copy (not a reference) — h is mutated by later layers.
		if cache.captureLayers != nil {
			for i, cl := range cache.captureLayers {
				if cl == l {
					cache.captured[i] = append(cache.captured[i][:0], h...)
				}
			}
		}
	}
	return h, nil
}

func addResidual(h, sub []float32) {
	n := min(len(h), len(sub))
	if n == 0 {
		return
	}
	h = h[:n]
	sub = sub[:n]
	_ = h[n-1]
	_ = sub[n-1]
	i := 0
	for ; i+3 < n; i += 4 {
		_ = h[i+3]
		_ = sub[i+3]
		h[i] += sub[i]
		h[i+1] += sub[i+1]
		h[i+2] += sub[i+2]
		h[i+3] += sub[i+3]
	}
	for ; i < n; i++ {
		h[i] += sub[i]
	}
}

func addResidual2(h, sub, sub2 []float32) {
	n := min(len(h), min(len(sub), len(sub2)))
	if n == 0 {
		return
	}
	h = h[:n]
	sub = sub[:n]
	sub2 = sub2[:n]
	_ = h[n-1]
	_ = sub[n-1]
	_ = sub2[n-1]
	i := 0
	for ; i+3 < n; i += 4 {
		_ = h[i+3]
		_ = sub[i+3]
		_ = sub2[i+3]
		h[i] += sub[i] + sub2[i]
		h[i+1] += sub[i+1] + sub2[i+1]
		h[i+2] += sub[i+2] + sub2[i+2]
		h[i+3] += sub[i+3] + sub2[i+3]
	}
	for ; i < n; i++ {
		h[i] += sub[i] + sub2[i]
	}
}

// normalizeInto applies the architecture's normalization from src into dst over one row:
// LayerNorm (mean-centered, with bias) for GPT-2/NeoX, else RMSNorm. bias is
// ignored by RMSNorm (and nil for the Sandwich4 post-norms).
func normalizeInto(arch *Architecture, dst, src, weight, bias []float32, dim int) {
	if arch.Norm == NormLayer {
		layerNormInto(dst, src, weight, bias, 1, dim, arch.NormEps)
		return
	}
	rmsNormInto(dst, src, weight, 1, dim, arch.NormEps, arch.RMSAddOne)
}

// normalize applies the architecture's normalization in place over one row.
func normalize(arch *Architecture, x, weight, bias []float32, dim int) {
	normalizeInto(arch, x, x, weight, bias, dim)
}

// forward runs runLayers then the final norm + LM head, returning the logit
// vector ([VocabSize]) for the next token. The head is the tied embedding
// (Gemma) or a separate lm_head (untied). Optional final
// logit soft-capping (Gemma 2; Gemma 3 = none).
func (m *Model) forward(id int, cache *KVCache) ([]float32, error) {
	h, err := m.runLayers(id, cache)
	if err != nil {
		return nil, err
	}
	if decodeTiming {
		t0 := time.Now()
		lg := m.logitsFromHidden(h, cache)
		atomic.AddInt64(&dtHead, int64(time.Since(t0)))
		return lg, nil
	}
	return m.logitsFromHidden(h, cache), nil
}

// ForwardCapture runs one forward for token id and returns the next-token logits plus the residual stream after each layer in
// `layers` (cloned): the read-only hidden-state seam a draft head reads (block drafters). The forward is byte-identical to
// forward(id); the captures are copies that never feed back. Layer indices are 0-based into [0, NumLayers); out[i]
// corresponds to layers[i].
//
// Every family's layers run through the program walker, which captures the residual after each layer (gate C of §4.8's
// seams PR holds it to the forward cut after that layer, bit for bit).
func (m *Model) ForwardCapture(id int, cache *KVCache, layers []int) (logits []float32, hidden [][]float32, err error) {
	_, done, err := m.enterCall(context.Background())
	if err != nil {
		return nil, nil, err
	}
	defer done()
	logits, hidden, err = m.forwardCapture(id, cache, layers)
	return logits, hidden, m.closeErr(context.Background(), err)
}

// forwardCapture is ForwardCapture without the lifecycle guard (lifecycle.go).
func (m *Model) forwardCapture(id int, cache *KVCache, layers []int) (logits []float32, hidden [][]float32, err error) {
	a := m.w.arch
	for _, l := range layers {
		if l < 0 || l >= a.NumLayers {
			return nil, nil, fmt.Errorf("decoder.ForwardCapture: layer %d out of range [0,%d)", l, a.NumLayers)
		}
	}
	cache.captureLayers = layers
	cache.captured = make([][]float32, len(layers))
	defer func() { cache.captureLayers, cache.captured = nil, nil }()
	lg, ferr := m.forward(id, cache)
	if ferr != nil {
		return nil, nil, ferr
	}
	return lg, cache.captured, nil
}

// ForwardSubCapture runs one token and returns, per layer, the attention contribution and the
// MLP contribution to the residual (scr.sub after each sublayer's sandwich-norm, before the add).
// This is the finer seam ForwardCapture doesn't give: it separates the two sublayers, which is
// what localizing a channel's sign flip to attention-vs-MLP at a specific layer requires.
// Diagnostic — same byte-identical-output contract as ForwardCapture. A single-operation block (Nemotron-H) fills one of the
// two; gate S of §4.8's seams PR rebuilds each layer's residual from them. Refused where a layer has an operation they do not
// cover (program.subCaptureUncovered).
func (m *Model) ForwardSubCapture(id int, cache *KVCache) (attn, mlp, ctx, mlpPre [][]float32, err error) {
	a := m.w.arch
	if m.program().subCaptureUncovered() {
		return nil, nil, nil, nil, fmt.Errorf("decoder.ForwardSubCapture: not wired for arch %q: a layer has an operation its contributions do not cover (Gemma 4's PLE branch or MoE block)", a.Name)
	}
	nL := a.NumLayers
	cache.subCapture = true
	cache.subAttn = make([][]float32, nL)
	cache.subMLP = make([][]float32, nL)
	cache.subMLPpre = make([][]float32, nL)
	cache.subCtx = make([][]float32, nL)
	defer func() {
		cache.subCapture = false
		cache.subAttn, cache.subMLP, cache.subMLPpre, cache.subCtx = nil, nil, nil, nil
	}()
	if _, ferr := m.forward(id, cache); ferr != nil {
		return nil, nil, nil, nil, ferr
	}
	return cache.subAttn, cache.subMLP, cache.subCtx, cache.subMLPpre, nil
}

// forwardFromEmbed is forward for a position whose residual-stream embedding is
// supplied directly (the multimodal embed-by-vector seam): it runs the layers
// from h and projects to logits, identical to forward(id) when h is that token's
// embedding.
func (m *Model) forwardFromEmbed(h []float32, cache *KVCache) ([]float32, error) {
	h, err := m.runLayersFromEmbed(h, cache)
	if err != nil {
		return nil, err
	}
	return m.logitsFromHidden(h, cache), nil
}

// softcapParallel applies Gemma's final-logit softcap sc*tanh(x/sc) in place, fanned out via parallelElementwise. Every
// element is independent and math.Tanh is deterministic, so splitting the loop is byte-identical to the serial form
// (TestSoftcapParallel_bitIdentical). It mirrors metal/model.go's softcapParallel.
func softcapParallel(logits []float32, softcap float32) {
	sc := softcap
	parallelElementwise(len(logits), func(lo, hi int) {
		for j := lo; j < hi; j++ {
			logits[j] = sc * float32(math.Tanh(float64(logits[j]/sc)))
		}
	})
}

// logitsFromHidden applies the final norm + LM head (tied embedding or separate
// lm_head) + optional Gemma logit soft-cap to a layer-stack output h, returning
// the next-token logits ([VocabSize]). One home for the head math, shared by
// forward and forwardFromEmbed.
func (m *Model) logitsFromHidden(h []float32, cache *KVCache) []float32 {
	hp := &m.program().head
	b := m.w.FinalNormBias
	if !hp.Norm.Bias {
		b = nil
	}
	applyNorm(hp.Norm, h, h, m.w.FinalNorm, b, m.w.arch.HiddenDim)
	return m.logitsFromNormed(h, cache)
}

// logitsFromNormed is logitsFromHidden for a hidden state that has ALREADY been through the final norm (the batched
// Qwen3.5 forward returns its rows normed): the LM head, softcap and logit scale, exactly as logitsFromHidden runs them.
func (m *Model) logitsFromNormed(h []float32, cache *KVCache) []float32 {
	hp := &m.program().head
	logits := cache.scr.logits // reused per stream; matmul fully overwrites it
	if hp.Tied {
		matmulInto(cache.scr.ws, m.be, &m.w.Embed, h, logits, 1) // tied: embedding doubles as the head
	} else {
		matmulInto(cache.scr.ws, m.be, &m.w.LMHead, h, logits, 1) // separate output projection
	}
	if hp.Softcap > 0 {
		softcapParallel(logits, float32(hp.Softcap))
	}
	if hp.LogitScale != 0 && hp.LogitScale != 1 { // Granite logits_scaling: logits /= scale
		inv := float32(1 / hp.LogitScale)
		n := len(logits)
		if n > 0 {
			_ = logits[n-1]
			i := 0
			for ; i+3 < n; i += 4 {
				_ = logits[i+3]
				logits[i] *= inv
				logits[i+1] *= inv
				logits[i+2] *= inv
				logits[i+3] *= inv
			}
			for ; i < n; i++ {
				logits[i] *= inv
			}
		}
	}
	return logits
}

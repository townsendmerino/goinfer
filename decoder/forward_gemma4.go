package decoder

import (
	"fmt"
	"math"
	"os"

	"github.com/townsendmerino/aikit/linalg"
)

var g4debug = os.Getenv("G4DEBUG") != "" // env-gated per-layer hidden-norm trace

// g4traceHidden, when non-nil, receives the residual stream after each layer
// (layer -1 = post-embedding) on every runLayersGemma4 call. Debug/test only:
// set it around the final token's forward to capture a last-position layer trace
// for diffing against the HF reference. The callback must copy h if it retains it.
var g4traceHidden func(layer int, h []float32)

// g4traceQKV, when non-nil, receives the post-norm/RoPE q, k, v for each layer on
// every runLayersGemma4 call (k/v nil on KV-shared layers). Debug/test only.
var g4traceQKV func(layer int, q, k, v []float32)

// runLayersGemma4 is the Gemma 4 (E-model) forward over one token. It diverges
// from the generic runLayers enough to warrant its own path: per-layer head_dim
// (256 local / 512 global), per-layer KV-head count, scale-less v-norm,
// cross-layer KV sharing (the last N layers reuse an earlier layer's KV),
// per-attention-type RoPE (proportional / partial-rotary on the global layers),
// the Per-Layer-Embedding (PLE) residual branch, and a per-layer output scalar.
// Returns the post-final-norm... no — returns the pre-final-norm hidden, matching
// runLayers; forward() applies the final norm + LM head + softcap (30).
//
// Buffers are allocated per call (parity-first; not the perf path).
func (m *Model) runLayersGemma4(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	g4 := arch.gemma4
	be := m.be
	hidden := arch.HiddenDim
	nH := arch.NumHeads
	pleDim := g4.HiddenSizePerLayerInput
	pos := cache.Pos()

	// Embedding × √hidden (inputs_embeds).
	h := make([]float32, hidden)
	m.w.Embed.Row(id, h)
	es := float32(arch.EmbedScale)
	for i := range h {
		h[i] *= es
	}
	if g4traceHidden != nil {
		g4traceHidden(-1, h)
	}

	// RoPE tables: local (full rotary, base 10k) and global (proportional /
	// partial-rotary, base 1e6 — only the first GlobalRotaryDim dims rotate).
	localInv := gemma4InvFreq(arch.HeadDim, arch.HeadDim, arch.RoPELocalBase)
	globalInv := gemma4InvFreq(g4.GlobalHeadDim, g4.GlobalRotaryDim, arch.RoPEGlobalBase)

	// Per-Layer-Embedding inputs: (token_identity + context_aware) / √2, shaped
	// [NumLayers, pleDim]. token_identity = per_layer_token_embd[id] × √pleDim;
	// context_aware = norm( per_layer_model_proj(inputs_embeds) × hidden^-0.5 ).
	perLayer := make([]float32, arch.NumLayers*pleDim)
	if pleDim > 0 {
		m.w.PerLayerTokenEmbed.Row(id, perLayer)
		tScale := float32(math.Sqrt(float64(pleDim)))
		for i := range perLayer {
			perLayer[i] *= tScale
		}
		ctxAware := make([]float32, arch.NumLayers*pleDim)
		matmul(be, &m.w.PerLayerModelProj, h, ctxAware, 1)
		cScale := float32(1.0 / math.Sqrt(float64(hidden)))
		for i := range ctxAware {
			ctxAware[i] *= cScale
		}
		inv2 := float32(1.0 / math.Sqrt2)
		for l := 0; l < arch.NumLayers; l++ {
			row := ctxAware[l*pleDim : (l+1)*pleDim]
			normalize(arch, row, m.w.PerLayerProjNorm, nil, pleDim) // RMSNorm over pleDim
			pl := perLayer[l*pleDim : (l+1)*pleDim]
			for i := range pl {
				pl[i] = (pl[i] + row[i]) * inv2
			}
		}
	}

	// Cross-layer KV sharing: layers ≥ firstShared reuse the KV of the last
	// non-shared layer of their type (sliding vs global).
	firstShared := arch.NumLayers - g4.SharedKVLayers
	lastSliding, lastGlobal := -1, -1
	for i := range firstShared {
		if arch.isGlobalLayer(i) {
			lastGlobal = i
		} else {
			lastSliding = i
		}
	}
	kvSrc := func(l int) int {
		if l < firstShared {
			return l
		}
		if arch.isGlobalLayer(l) {
			return lastGlobal
		}
		return lastSliding
	}

	// Attention scratch: allocated once per token, reused across layers (the
	// batched-over-heads gemma4Attend gathers K_head/V_headᵀ + per-head scores).
	// Sized to the widest per-layer head_dim and the current key count.
	maxHd := max(g4.GlobalHeadDim, arch.HeadDim)
	nkMax := pos + 1
	g4sc := g4attnScratch{
		kh:     make([]float32, nkMax*maxHd),
		vt:     make([]float32, nkMax*maxHd),
		qstack: make([]float32, nH*maxHd),
		scores: make([]float32, nH*nkMax),
		cstack: make([]float32, nH*maxHd),
	}

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]
		global := arch.isGlobalLayer(l)
		hd := arch.headDimAt(l)
		nKV := arch.kvHeadsAt(l)
		ffn := arch.ffnAt(l)
		invFreq := localInv
		if global {
			invFreq = globalInv
		}

		// --- attention sub-block (sandwich) ---
		normd := make([]float32, hidden)
		copy(normd, h)
		normalize(arch, normd, lw.PreAttnNorm, nil, hidden)

		q := make([]float32, nH*hd)
		matmul(be, &lw.QProj, normd, q, 1)
		rmsNorm(q, lw.QNorm, nH, hd, arch.NormEps, arch.RMSAddOne)
		applyRoPE(q, nH, hd, pos, invFreq, 1.0)

		if l < firstShared { // owns its KV
			k := make([]float32, nKV*hd)
			matmul(be, &lw.KProj, normd, k, 1)
			v := make([]float32, nKV*hd)
			if lw.VFromK { // attention_k_eq_v: V is v_norm(k_proj output)
				copy(v, k)
			} else {
				matmul(be, &lw.VProj, normd, v, 1)
			}
			rmsNorm(k, lw.KNorm, nKV, hd, arch.NormEps, arch.RMSAddOne) // K: k_norm + RoPE
			applyRoPE(k, nKV, hd, pos, invFreq, 1.0)
			rmsNormNoWeight(v, nKV, hd, arch.NormEps) // V: scale-less v_norm, no RoPE
			cache.Append(l, k, v)
			if g4traceQKV != nil {
				g4traceQKV(l, q, k, v)
			}
		} else if g4traceQKV != nil {
			g4traceQKV(l, q, nil, nil)
		}
		src := kvSrc(l)
		keys, vals := cache.Keys(src), cache.Vals(src)
		kvDim := nKV * hd
		nKeys := len(keys) / kvDim
		ctx := make([]float32, nH*hd)
		start := cache.WindowStart(pos, global)
		gemma4Attend(q, ctx, keys, vals, nH, nKV, hd, start, nKeys, arch.AttnScale, &g4sc)

		attnOut := make([]float32, hidden)
		matmul(be, &lw.OProj, ctx, attnOut, 1)
		normalize(arch, attnOut, lw.PostAttnNorm, nil, hidden) // post-attn (sandwich)
		for i := range h {
			h[i] += attnOut[i]
		}

		// --- FFN sub-block ---
		if lw.gemma4moe != nil {
			// Gemma 4 26B-A4B: parallel dense+MoE FFN (enable_moe_block). gemma4MoEFFN
			// returns the FULL layer output — post-attention residual h + the joint-normed
			// (dense ‖ MoE) branches, × layer_scalar — so it replaces the dense MLP + PLE +
			// separate scalar tail below. (The MoE variant is PLE-free.)
			h = gemma4MoEFFN(be, arch, h, lw.gemma4moe, m.pager)
		} else {
			// dense variant: MLP sub-block (sandwich), GeGLU at the per-layer width.
			copy(normd, h)
			normalize(arch, normd, lw.PreMLPNorm, nil, hidden)
			gate := make([]float32, ffn)
			up := make([]float32, ffn)
			matmul(be, &lw.GateProj, normd, gate, 1)
			matmul(be, &lw.UpProj, normd, up, 1)
			parallelElementwise(len(gate), func(lo, hi int) {
				for i := lo; i < hi; i++ {
					gate[i] = geluTanh(gate[i]) * up[i]
				}
			})
			mlpOut := make([]float32, hidden)
			matmul(be, &lw.DownProj, gate, mlpOut, 1)
			normalize(arch, mlpOut, lw.PostMLPNorm, nil, hidden) // post-FFN (sandwich)
			for i := range h {
				h[i] += mlpOut[i]
			}

			// --- PLE branch: gate→gelu→×per-layer-embedding→proj→norm→+residual ---
			if pleDim > 0 {
				pl := perLayer[l*pleDim : (l+1)*pleDim]
				px := make([]float32, pleDim)
				matmul(be, &lw.PLEGate, h, px, 1)
				parallelElementwise(len(px), func(lo, hi int) {
					for i := lo; i < hi; i++ {
						px[i] = geluTanh(px[i]) * pl[i]
					}
				})
				pout := make([]float32, hidden)
				matmul(be, &lw.PLEProj, px, pout, 1)
				normalize(arch, pout, lw.PostPLENorm, nil, hidden)
				for i := range h {
					h[i] += pout[i]
				}
			}

			// --- per-layer output scalar ---
			if lw.LayerScalar != 0 {
				for i := range h {
					h[i] *= lw.LayerScalar
				}
			}
		}
		if g4debug {
			var ss float64
			for _, e := range h {
				ss += float64(e) * float64(e)
			}
			kind := "loc"
			if global {
				kind = "GLB"
			}
			vk := ""
			if lw.VFromK {
				vk = " kv=eq"
			}
			fmt.Printf("  L%-2d %s hd=%d nkv=%d scale=%.4f ||h||=%.3f%s\n", l, kind, hd, nKV, lw.LayerScalar, math.Sqrt(ss), vk)
		}
		if g4traceHidden != nil {
			g4traceHidden(l, h)
		}
		cache.captureResidual(l, h)
	}
	cache.Advance()
	return h, nil
}

// gemma4InvFreq builds the RoPE inverse-frequency table. For full rotary,
// rotaryDim == headDim and the result is headDim/2 entries. For proportional
// (partial) rotary, rotaryDim < headDim: the first rotaryDim/2 entries are the
// real frequencies (spaced by 1/headDim, NOT 1/rotaryDim) and the rest are zero,
// so applyRoPE (which rotates len(invFreq) pairs split at headDim/2) leaves the
// trailing dims unrotated — matching HF's "proportional" RoPE.
func gemma4InvFreq(headDim, rotaryDim int, base float64) []float64 {
	half := headDim / 2
	inv := make([]float64, half)
	angles := rotaryDim / 2
	for i := range angles {
		inv[i] = 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
	}
	return inv // entries [angles:half] stay 0 → no rotation there
}

// rmsNormNoWeight applies a scale-less RMSNorm per head (Gemma 4's v_norm:
// with_scale=False — normalize V by RMS, no learned weight).
func rmsNormNoWeight(x []float32, rows, dim int, eps float64) {
	for r := range rows {
		v := x[r*dim : r*dim+dim]
		var ss float64
		for _, e := range v {
			ss += float64(e) * float64(e)
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(dim)+eps))
		for i := range v {
			v[i] *= inv
		}
	}
}

// g4attnScratch holds the batched-over-heads gemma4 attention buffers: the
// gathered K_head [n,hd] / V_headᵀ [hd,n] for one KV head, the stacked query-head
// group, its scores, and the context. Allocated once per token in
// runLayersGemma4 (sized to the widest head_dim and current key count), reused
// across layers — replaces the per-layer-per-token `scores := make`.
type g4attnScratch struct {
	kh, vt, qstack, scores, cstack []float32
}

// gemma4Attend computes one token's grouped-query attention over keys
// [start, nKeys), with explicit per-layer dims (head_dim / KV-head count vary by
// layer). It batches the query heads sharing a KV head into one A·Bᵀ matmul: the
// token is a single query position, so every head attends the same key range
// (no causal mask). Per KV head — gather K_head [n,hd] + V_headᵀ [hd,n] once,
// QKᵀ = MatmulBT(group_q, K_head), a scaled softmax per row, scores·V =
// MatmulBT(scores, V_headᵀ), scatter into ctx. This moves the QKᵀ/scores·V
// scalar triple-loops (the per-token O(L²) prefill cost) onto the SIMD kernel.
// scale is arch.AttnScale (1.0 for gemma4 — the q/k-norm weights absorb it).
// Parity is argmax + cosine, not bit-identical (float64→f32 SIMD).
func gemma4Attend(q, ctx, keys, vals []float32, nH, nKV, hd, start, nKeys int, scale float64, sc *g4attnScratch) {
	kvDim := nKV * hd
	group := nH / nKV
	n := nKeys - start
	for kvh := range nKV {
		kh, vt := sc.kh[:n*hd], sc.vt[:hd*n]
		for j := range n {
			base := (start+j)*kvDim + kvh*hd
			copy(kh[j*hd:j*hd+hd], keys[base:base+hd])
			vrow := vals[base : base+hd]
			for d := range hd {
				vt[d*n+j] = vrow[d]
			}
		}
		qs := sc.qstack[:group*hd]
		for g := range group {
			qhead := kvh*group + g
			copy(qs[g*hd:g*hd+hd], q[qhead*hd:qhead*hd+hd])
		}
		// QKᵀ: scores[group,n] = group_q[group,hd] · K_head[n,hd]ᵀ
		scores := sc.scores[:group*n]
		linalg.MatmulBT(qs, kh, scores, group, hd, n)
		for g := range group {
			rowS := scores[g*n : g*n+n]
			maxS := math.Inf(-1)
			for j := range n {
				v := float64(rowS[j]) * scale
				rowS[j] = float32(v)
				if v > maxS {
					maxS = v
				}
			}
			var sum float64
			for j := range n {
				e := math.Exp(float64(rowS[j]) - maxS)
				rowS[j] = float32(e)
				sum += e
			}
			inv := 1.0 / sum
			for j := range n {
				rowS[j] = float32(float64(rowS[j]) * inv)
			}
		}
		// scores·V: ctx_group[group,hd] = scores[group,n] · V_head[n,hd]
		//                               = MatmulBT(scores, V_headᵀ[hd,n])
		cs := sc.cstack[:group*hd]
		linalg.MatmulBT(scores, vt, cs, group, n, hd)
		for g := range group {
			qhead := kvh*group + g
			copy(ctx[qhead*hd:qhead*hd+hd], cs[g*hd:g*hd+hd])
		}
	}
}
